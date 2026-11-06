"""Development-only lab replay through real search, ColorTracker and Mission.

Goal takes contain a mislabeled, point-up yellow triangle: flip only those frames.
No camera, transport, training, model updates, or flight qualification is involved.
"""

import argparse
from collections import Counter
import hashlib
import itertools
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from dtr.flight import ColorTracker, Mission
from dtr.flight_pipeline import FlightPipeline
from dtr.goal_square import DEFAULT_MAX_CROPS, find_goals, goal_candidates
from dtr.runtime import Predictor
from dtr.search_worker import DEFAULT_BALLOON_LIMIT
from dtr.vision import GREEN_PURPLE_PROFILE, observe


class SimulatedWorker:
    """One in-flight search; delivery and resubmission use the camera clock.

    Computing search synchronously here does not advance that clock. poll() only
    exposes its result on the first camera frame at/after the completion time.
    """

    def __init__(self, search, latency, dispatch_s=0.0):
        if not np.isfinite(dispatch_s) or dispatch_s < 0:
            raise ValueError("Dispatch overhead must be finite and nonnegative")
        self.search, self.latency, self.dispatch_s = search, latency, dispatch_s
        self.now, self.pending = 0.0, None
        self.submissions, self.deliveries = [], []

    def submit(self, rgb, frame_time, phase):
        if self.pending is not None:
            return False
        detections, crops = self.search(rgb, phase)
        delay = float(self.latency(phase, crops))
        if not np.isfinite(delay) or delay < 0:
            raise ValueError("Latency must be finite and nonnegative")
        # The worker's search_ms timer starts only after the pipeline step and frame
        # hand-off; that dispatch stage delays availability but not the result's age origin.
        self.pending = (frame_time + self.dispatch_s + delay, dict(frame_time=frame_time, phase=phase,
                        detections=detections, search_ms=delay * 1000, neural_calls=crops))
        self.submissions.append(dict(frame_time=frame_time, latency_s=delay, crops=crops))
        return True

    def poll(self):
        if self.pending is None or self.now < self.pending[0]:
            return None
        _, result = self.pending
        self.pending = None
        self.deliveries.append(dict(submitted=result['frame_time'], delivered=self.now))
        return result


def load_take(path):
    scene = json.loads((path / 'scene.json').read_text())
    rows = [json.loads(line) for line in (path / 'frames.jsonl').read_text().splitlines()]
    rows = [row for row in rows if row.get('raw')]
    times = [float(row['frame_time']) for row in rows]
    if not times or not np.isfinite(times).all() or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError(f"Need increasing finite frame times: {path}")
    return dict(path=path, scene=scene, rows=rows)


def read_rgb(take, row, flip=False):
    path = (take['path'] / row['raw']).resolve()
    if not path.is_relative_to((take['path'] / 'raw').resolve()) or path.name.startswith('IMG_'):
        raise ValueError(f"Not a lab raw frame: {path}")
    bgr = cv2.imread(str(path))
    if bgr is None or bgr.shape != (240, 320, 3):
        raise ValueError(f"Expected a 320x240 RGB lab frame: {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return np.ascontiguousarray(rgb[::-1]) if flip else rgb


def estimate_dispatch_ms(takes, limit_ms=200):
    """Constant frame-to-worker overhead that best reproduces logged delivery frames.

    A result logged on frame i became available after frame i-1 and by frame i, so
    overhead d is consistent when t[i-1] < submit + d + search_ms <= t[i]. Choose the
    whole millisecond matching most deliveries; ties prefer fewer early deliveries.
    """
    bounds = []
    for take in takes:
        rows = take['rows']
        for previous, row in zip(rows, rows[1:]):
            result = row.get('search_result')
            if result is None:
                continue
            ready = float(result['frame_time']) + float(result['search_ms']) / 1000
            bounds.append(((float(previous['frame_time']) - ready) * 1000,
                           (float(row['frame_time']) - ready) * 1000))
    if not bounds:
        return dict(dispatch_ms=0.0, samples=0, exact=0, early=0, late=0)
    lower, upper = np.array(bounds).T
    scored = []
    for d in range(limit_ms + 1):
        exact, early = int(((d > lower) & (d <= upper)).sum()), int((d <= lower).sum())
        scored.append((exact, -early, -d, d))
    exact, early, _, d = max(scored)
    return dict(dispatch_ms=float(d), samples=len(bounds), exact=exact, early=-early,
                late=len(bounds) - exact + early)


def fit_cost_model(takes, logged_balloon_limit=8):
    """Fit shared fixed ms + phase-specific ms/crop, constrained nonnegative.

    Count goal crops on the original (unflipped) submission, not its delivery
    frame. These lab logs used all eight balloon slots (verified by the before
    replay); never recount them with the NEW proposal filter. Exclude submissions
    outside the recording. --cost-model also freezes this fit across code changes.
    """
    samples, excluded = [], 0
    for take in takes:
        rows = take['rows']
        times = np.array([r['frame_time'] for r in rows])
        for row in rows:
            result = row.get('search_result')
            if result is None:
                continue
            index = int(np.argmin(abs(times - result['frame_time'])))
            if abs(times[index] - result['frame_time']) > 1e-6:
                excluded += 1
                continue
            phase = result['phase']
            rgb = read_rgb(take, rows[index])
            count = (min(DEFAULT_MAX_CROPS, len(goal_candidates(rgb))) if phase == 'goal'
                     else logged_balloon_limit)
            samples.append(dict(take=take['path'].name, phase=phase, crops=count,
                                search_ms=float(result['search_ms'])))
    if not samples:
        raise ValueError('No logged searches match recorded submission frames; supply --cost-model')
    x = np.array([[1, s['crops'] if s['phase'] == 'goal' else 0,
                   s['crops'] if s['phase'] == 'balloon' else 0] for s in samples], float)
    y = np.array([s['search_ms'] for s in samples])
    if np.linalg.matrix_rank(x) < 3:
        raise ValueError('Cost fit is underdetermined; fit all takes or supply --cost-model')
    fits = []
    for size in range(1, 4):
        for columns in itertools.combinations(range(3), size):
            values = np.linalg.lstsq(x[:, columns], y, rcond=None)[0]
            if (values >= 0).all():
                coefficients = np.zeros(3)
                coefficients[list(columns)] = values
                fits.append((float(np.sum((x @ coefficients - y) ** 2)), coefficients))
    coefficients = min(fits, key=lambda item: item[0])[1]
    predicted = x @ coefficients
    factors = {}
    for name in sorted({s['take'] for s in samples}):
        indices = [i for i, s in enumerate(samples) if s['take'] == name]
        factors[name] = float(np.median(y[indices] / np.maximum(predicted[indices], 1e-9)))
    diagnostics = {}
    for phase in ('goal', 'balloon'):
        indices = [i for i, s in enumerate(samples) if s['phase'] == phase]
        residual = predicted[indices] - y[indices]
        scaled = np.array([predicted[i] * factors[samples[i]['take']] for i in indices])
        diagnostics[phase] = dict(samples=len(indices),
            crop_histogram=dict(Counter(samples[i]['crops'] for i in indices)),
            rmse_ms=float(np.sqrt(np.mean(residual ** 2))),
            calibrated_rmse_ms=float(np.sqrt(np.mean((scaled - y[indices]) ** 2))),
            measured_median_ms=float(np.median(y[indices])))
    return dict(fixed_ms=float(coefficients[0]),
                per_crop_ms=dict(goal=float(coefficients[1]), balloon=float(coefficients[2])),
                take_factors=factors, diagnostics=diagnostics, samples=samples,
                excluded_unrecorded_submissions=excluded,
                logged_balloon_crops=logged_balloon_limit,
                scope='Estimated Pi search latency, not new Pi measurements. Shared fixed cost; '
                      'per-take median multiplicative correction for observed scheduling load. '
                      'Goal crop counts reconstructed with current goal search on unflipped frames. '
                      'No residual jitter simulation; short takes are sensitive to delivery timing.')


def search_frame(rgb, phase, predictors, balloon_limit):
    """The same search/admission semantics as search_worker.ModelSearch."""
    if phase == 'goal':
        found = find_goals(rgb, predictors[phase])
        detections, crops = found['detections'], found['neural_calls']
    else:
        found = observe(rgb, predictors['balloon'], profile=GREEN_PURPLE_PROFILE, limit=balloon_limit)
        detections = [d for d in found['observations'] if d['accepted'] and not d['suppressed']]
        crops = len(found['observations'])
    return ([dict(label=d['label'], box=list(map(float, d['box'])), score=float(d['score']))
             for d in detections], crops)


def replay_take(take, predictors, cost, balloon_limit, latency, output):
    phase = take['scene']['phase']
    expected = 'yellow_triangle' if phase == 'goal' else 'green_balloon'
    frames = [read_rgb(take, row, flip=phase == 'goal') for row in take['rows']]
    # Search every frame once for independent per-frame recall, then replay those
    # actual search results only when the single-slot simulated worker submits.
    searches = [search_frame(rgb, phase, predictors, balloon_limit) for rgb in frames]
    index = 0

    def search(rgb, requested_phase):
        if requested_phase != phase:
            return search_frame(rgb, requested_phase, predictors, balloon_limit)
        return searches[index]

    def delay(requested_phase, crops):
        if latency is not None:
            return latency
        cost_phase = 'goal' if requested_phase == 'goal' else 'balloon'
        return (cost['fixed_ms'] + cost['per_crop_ms'][cost_phase] * crops) * (
            cost['take_factors'].get(take['path'].name, 1.0)) / 1000

    worker = SimulatedWorker(search, delay, cost.get('dispatch_ms', 0.0) / 1000)
    mission = Mission('yellow')
    if phase == 'goal':
        mission.captured()
    pipeline = FlightPipeline(worker, ColorTracker(), mission)
    results, ages, commands, wrong = [], [], Counter(), Counter()
    fresh, ready_without_size_center, expected_frames = 0, 0, []
    for index, (rgb, recorded) in enumerate(zip(frames, take['rows'])):
        now = float(recorded['frame_time'])
        worker.now = now
        row = pipeline.step(rgb, now)
        # LOST and committed-ENTER frames repeat a frozen target with no live track.
        target = row['message']['target'] if row['selected_track_id'] is not None else None
        commands[row['message']['cmd']] += 1
        selected = next((t for t in row['tracks'] if t['id'] == row['selected_track_id']), None)
        if target:
            ages.append(target['age_ms'])
            if target['label'] == expected:
                expected_frames.append(index)
            else:
                wrong[target['label']] += 1
            track = next(t for t in pipeline.tracker.tracks if t.id == row['selected_track_id'])
            is_fresh = mission.freshness(target, track)
            fresh += is_fresh
            track = next(t for t in pipeline.tracker.tracks if t.id == row['selected_track_id'])
            grown = all(a <= mission.config.max_growth * b for a, b in zip(
                (track.box[2] - track.box[0], track.box[3] - track.box[1]), track.confirmed_wh))
            ready_without_size_center += bool(is_fresh and grown)
        results.append(dict(frame=recorded['raw'], frame_time=now,
                            message=row['message'], selected=selected,
                            search_result=row['search_result'], submitted=row['submitted']))
    first = expected_frames[0] if expected_frames else None
    offline = sum(any(d['label'] == expected for d in ds) for ds, _ in searches)
    offline_wrong = Counter(d['label'] for ds, _ in searches for d in ds if d['label'] != expected)
    name = take['path'].name
    # Evenly spaced snapshots plus first detection, each with its selected box.
    indices = sorted(set(np.linspace(0, len(frames) - 1, 12, dtype=int).tolist()
                         + ([first] if first is not None else [])))
    sheet = Image.new('RGB', (4 * 320, ((len(indices) + 3) // 4) * 270), '#18222b')
    for slot, i in enumerate(indices):
        tile = Image.fromarray(frames[i])
        draw = ImageDraw.Draw(tile)
        selected = results[i]['selected']
        if selected:
            draw.rectangle(selected['box'], outline='lime' if selected['label'] == expected else 'red', width=2)
            draw.text((4, 4), selected['label'], fill='white', stroke_width=1, stroke_fill='black')
        x, y = slot % 4 * 320, slot // 4 * 270
        sheet.paste(tile, (x, y))
        ImageDraw.Draw(sheet).text((x + 4, y + 242),
            f"{i}: {results[i]['message']['cmd']} age={results[i]['message']['target']['age_ms'] if selected else '-'}ms",
            fill='white')
    contact = output / f'{name}.png'
    sheet.save(contact)
    return dict(take=name, phase=phase, distance_m=take['scene']['distance_m'], expected=expected,
        flipped=phase == 'goal', frames=len(frames), expected_selected_frames=len(expected_frames),
        fresh_s=mission.config.fresh_s, confirm_timeout_s=pipeline.tracker.confirm_timeout,
        share_after_first_detection=(len(expected_frames) / (len(frames) - first) if first is not None else None),
        time_to_first_detection_s=(results[first]['frame_time'] - results[0]['frame_time'] if first is not None else None),
        target_age_ms=dict(min=min(ages) if ages else None, median=float(np.median(ages)) if ages else None),
        freshness_frames=int(fresh), ready_ignoring_size_center_frames=int(ready_without_size_center),
        command_counts=dict(commands), wrong_label_selections=dict(wrong),
        offline_expected_frames=offline, offline_wrong_detections=dict(offline_wrong),
        crop_histogram=dict(Counter(c for _, c in searches)),
        simulated_latency_ms=dict(min=min(s['latency_s'] for s in worker.submissions) * 1000,
            median=float(np.median([s['latency_s'] for s in worker.submissions])) * 1000),
        submissions=worker.submissions, deliveries=worker.deliveries, contact_sheet=str(contact), rows=results)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('takes', nargs='*', type=Path)
    parser.add_argument('--models', type=Path, default=Path.home() /
                        'Downloads/Find Target Lab Kit - 2026-11-02/Pi Bundle/models')
    parser.add_argument('--output', type=Path, required=True, help='New directory below /tmp')
    parser.add_argument('--balloon-limit', type=int, default=DEFAULT_BALLOON_LIMIT)
    parser.add_argument('--latency', type=float, help='Fixed search latency in seconds, overriding fit')
    parser.add_argument('--cost-model', type=Path, help='Freeze fit from a previous replay JSON')
    parser.add_argument('--dispatch-ms', type=float,
                        help='Frame-to-worker overhead before search_ms starts; default fits the logs')
    args = parser.parse_args()
    if args.latency is not None and (not np.isfinite(args.latency) or args.latency < 0):
        parser.error('--latency must be finite and nonnegative')
    if args.dispatch_ms is not None and (not np.isfinite(args.dispatch_ms) or args.dispatch_ms < 0):
        parser.error('--dispatch-ms must be finite and nonnegative')
    if not 1 <= args.balloon_limit <= 64:
        parser.error('--balloon-limit must be between 1 and 64')
    output = args.output.resolve()
    if not output.is_relative_to(Path('/tmp').resolve()) or output == Path('/tmp').resolve():
        parser.error('--output must be a new directory below /tmp')
    output.mkdir(parents=True, exist_ok=False)
    paths = args.takes or sorted(Path.home().glob('Desktop/2026110[23]T*-*'))
    takes = [load_take(path.expanduser()) for path in paths]
    if args.cost_model:
        cost = json.loads(args.cost_model.read_text())['latency_model']
    elif args.latency is not None:
        cost = dict(scope='Fixed latency override; no fit', fixed_ms=args.latency * 1000,
                    per_crop_ms=dict(goal=0, balloon=0), take_factors={}, samples=[])
    else:
        cost = fit_cost_model(takes)
    if args.dispatch_ms is not None:
        cost['dispatch_ms'] = args.dispatch_ms
    elif 'dispatch_ms' not in cost:
        cost['dispatch_fit'] = estimate_dispatch_ms(takes)
        cost['dispatch_ms'] = cost['dispatch_fit']['dispatch_ms']
    print(json.dumps({k: v for k, v in cost.items() if k != 'samples'}, indent=2), flush=True)
    predictors = {phase: Predictor(args.models / f'{phase}.tflite', allow_unvalidated=True)
                  for phase in ('goal', 'balloon')}
    report = dict(scope='Development evaluation only, never training or flight qualification',
                  initial_state='Ready worker, empty tracks, no inherited in-flight result per take',
                  balloon_limit=args.balloon_limit, latency_override_s=args.latency, latency_model=cost,
                  model_sha256={phase: p.metadata['sha256'] for phase, p in predictors.items()}, takes=[])
    root = Path(__file__).resolve().parents[1]
    sources = ('scripts/replay_lab_sessions.py', 'src/dtr/flight.py', 'src/dtr/flight_pipeline.py',
               'src/dtr/vision.py', 'src/dtr/search_worker.py', 'src/dtr/goal_square.py')
    report['source_sha256'] = {path: hashlib.sha256((root / path).read_bytes()).hexdigest()
                               for path in sources}
    for take in takes:
        result = replay_take(take, predictors, cost, args.balloon_limit, args.latency, output)
        report['takes'].append(result)
        (output / 'replay.json').write_text(json.dumps(report, indent=2) + '\n')
        print(f"{result['take']}: offline {result['offline_expected_frames']}/{result['frames']}; "
              f"selected {result['expected_selected_frames']}; fresh {result['freshness_frames']}; "
              f"wrong {result['wrong_label_selections']}", flush=True)


if __name__ == '__main__':
    main()
