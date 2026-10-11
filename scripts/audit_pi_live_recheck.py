"""Recount sequential camera smoke checks; neither an accuracy nor speed qualification."""
import json
from pathlib import Path

from dtr.data import sha256, write_json
from scripts.review_balloon_live import review
from scripts.pi_balloon_live_schedule_bench import trial_summary
from scripts.review_scheduled_balloon_search import validate_schedule

OUTPUT = Path('research/balloon-red-blue-20261008/pi-live-recheck-20261011-1423.json')
MODEL_SHA = 'b80460a7526173218b761b265176d19b576d0fa24be14250bf2423e0ad607530'
RUNTIME_SHA = '54ed7fdda0b5b5f6db05a538eb800b7bc7ecd70d32f67dc13d1e78817d8b653a'


def audit():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    results, intervals, identity = {}, [], None
    for policy in ('full', 'periodic'):
        root = Path(f'runs/balloon-live-{policy}-20261011-1423')
        checked = review(root)
        report = checked['report']
        rows = [json.loads(line) for line in (root / 'frames.jsonl').read_text().splitlines()]
        fresh = {**validate_schedule(policy, rows), **trial_summary(rows, report['elapsed_s'])}
        if (any(report[k] != v for k, v in fresh.items()) or report['policy'] != policy
                or report['model_sha256'] != MODEL_SHA or report['runtime_library_sha256'] != RUNTIME_SHA
                or report['script_sha256'] != sha256('scripts/pi_balloon_live_schedule_bench.py')
                or report['live_helper_sha256'] != sha256('scripts/pi_balloon_live_bench.py')
                or report['max_pre_camera_golden_score_delta'] > .001
                or not checked['timing_validation']['temporal_observation_valid']):
            raise ValueError('Changed runtime or invalid raw camera evidence')
        current = tuple(report[k] for k in ('model_sha256', 'runtime_library_sha256',
            'region_library_sha256', 'bundle_sha256', 'scheduler_sha256', 'requested_camera_fps'))
        if identity is not None and current != identity:
            raise ValueError('Different policy artifacts or capture requests')
        identity = current
        intervals.append((rows[0]['capture_started_boottime_ns'], rows[-1]['result_boottime_ns']))
        results[policy] = dict(
            path=str(root), report_sha256=checked['report_sha256'],
            frames=report['frames'], observed_live_fps=report['observed_live_fps'],
            sensor_to_result=report['sensor_to_result'], full_observation_age=report['full_observation_age'],
            mode_counts=report['mode_counts'], mean_neural_calls=report['mean_neural_calls'],
            accepted_counts=report['accepted_counts'], health_before=report['health_before'],
            health_after=report['health_after'], timing_validation=checked['timing_validation'],
            sample_health=checked['samples'],
            visually_reviewed_samples={f'{i:04d}-raw.png': report['files'][f'{i:04d}-raw.png'] for i in (0, 59)},
            visually_reviewed_detection_sample={'0001-annotated.png': report['files']['0001-annotated.png']})
    if intervals[0][1] >= intervals[1][0]:
        raise ValueError('Timed camera trials overlap')
    write_json(OUTPUT, dict(
        runs=results, timed_intervals_boottime_ns=intervals, camera_trials_nonoverlapping=True,
        model_sha256=MODEL_SHA, runtime_library_sha256=RUNTIME_SHA,
        script_sha256=sha256(__file__), helpers_sha256={p: sha256(p) for p in (
            'scripts/review_balloon_live.py', 'scripts/pi_balloon_live_schedule_bench.py',
            'scripts/review_scheduled_balloon_search.py')},
        visual_finding='Main agent inspected first/last raw images and frame1 overlay in both runs. Dim room with blue-tinted window/curtain at left; no visible balloons. Frame1 has an erroneous blue-balloon label on a tiny clipped bottom-left patch.',
        limitations=['Only bounded sample images retained; unsaved frames have no visual labels.',
            'One sequential run per policy, no counterbalanced repetitions or representative target workload.',
            'File transfer occurred in the broader session; no all-process CPU isolation claim.',
            'This establishes current camera/runtime execution, not competition accuracy or a guaranteed FPS.'],
        camera_sensor_size=[2592, 1944], processing_size=[320, 240], camera_rotation_degrees=180,
        model_replaced=False, esp32_commands_sent=False, motor_commands_sent=False,
        test_evaluated=False, accuracy_measured=False, deployment_approved=False,
        qualification_achieved=False))
    print({k: {key: r[key] for key in ('frames', 'observed_live_fps', 'sensor_to_result', 'accepted_counts')} for k, r in results.items()})


if __name__ == '__main__':
    audit()
