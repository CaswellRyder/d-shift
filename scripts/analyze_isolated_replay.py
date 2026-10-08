"""Analyze saved Pi receipts without loading a model or accessing hardware.

Pixel size is not physical distance. Only yellow localization is scored.
Paired replay results use frames processed by BOTH methods, not skipped frames.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path


def matched_truth(predictions, truth):
    """Maximum-cardinality one-to-one localization matches at IoU >= 0.5."""
    def iou(a, b):
        overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
            0, min(a[3], b[3]) - max(a[1], b[1]))
        union = ((a[2] - a[0]) * (a[3] - a[1])
                 + (b[2] - b[0]) * (b[3] - b[1]) - overlap)
        return overlap / union if union > 0 else 0

    edges = [[j for j, box in enumerate(truth) if iou(pred, box) >= .5]
             for pred in predictions]
    owners = {}

    def visit(index, seen):
        for target in edges[index]:
            if target in seen:
                continue
            seen.add(target)
            if target not in owners or visit(owners[target], seen):
                owners[target] = index
                return True
        return False

    for index in range(len(predictions)):
        visit(index, set())
    return set(owners)


def score(rows):
    buckets = {key: dict(hits=0, targets=0) for key in
               ('under16', '16to32', '32to64', '64plus')}
    totals = dict(tp=0, fp=0, fn=0, target_frames=0, hit_frames=0)
    for row in rows:
        truth = [item['box'] for item in row['truth']
                 if item['label'].startswith('yellow_')]
        matched = matched_truth(row['boxes'], truth)
        totals['tp'] += len(matched)
        totals['fp'] += len(row['boxes']) - len(matched)
        totals['fn'] += len(truth) - len(matched)
        totals['target_frames'] += bool(truth)
        totals['hit_frames'] += bool(matched)
        for index, box in enumerate(truth):
            size = math.sqrt(max(0, box[2] - box[0]) * max(0, box[3] - box[1]))
            key = ('under16' if size < 16 else '16to32' if size < 32
                   else '32to64' if size < 64 else '64plus')
            buckets[key]['targets'] += 1
            buckets[key]['hits'] += index in matched
    for bucket in buckets.values():
        bucket['recall'] = (bucket['hits'] / bucket['targets']
                            if bucket['targets'] else None)
    return dict(frames=len(rows), **totals, size_buckets=buckets)


def pair_rows(pixel, context):
    def indexed(rows):
        result = {}
        for row in rows:
            key = (row['source'], row['clip'], row['frame'])
            if key in result:
                raise ValueError('Duplicate replay frame')
            result[key] = row
        return result
    p, c = indexed(pixel), indexed(context)
    keys = sorted(p.keys() & c.keys())
    for key in keys:
        if p[key]['truth'] != c[key]['truth']:
            raise ValueError('Paired ground truth differs')
    return [p[key] for key in keys], [c[key] for key in keys]


def reacquisition(rows, fps=10, return_frame=13):
    """First correct delivered result after known generated blackout (10..12).

    Not a physical-camera measurement; includes skipped frames and result age.
    No-hit clips remain explicit rather than disappearing from the average.
    """
    clips = {}
    for row in rows:
        clips.setdefault((row['source'], row['clip']), []).append(row)
    results = []
    for (source, clip), sequence in sorted(clips.items()):
        returning = sorted((r for r in sequence if r['frame'] >= return_frame),
                           key=lambda r: r['frame'])
        target_rows = [r for r in returning if any(
            t['label'].startswith('yellow_') for t in r['truth'])]
        hits = [r for r in target_rows if score([r])['tp']]
        delay = min(((r['frame'] - return_frame) * 1000 / fps
                     + r['result_age_ms'] for r in hits), default=None)
        results.append(dict(source=source, clip=clip,
                            target_observed_after_return=bool(target_rows),
                            reacquired=bool(hits), delivery_delay_ms=delay))
    return results


def scheduling(rows, frames_per_clip=30):
    scheduled = len({(r['source'], r['clip']) for r in rows}) * frames_per_clip
    return dict(scheduled=scheduled, processed=len(rows),
                total_skipped=scheduled - len(rows),
                skipped_before_processed_frames=sum(r['skipped'] for r in rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('results', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = dict(scope='yellow development replay; not competition or physical range',
                  size_definition='sqrt(box area), pixels at 320x240', sources={}, passes={})

    def load(name):
        path = args.results / name / 'frames.jsonl'
        raw = path.read_bytes()
        output['sources'][name] = hashlib.sha256(raw).hexdigest()
        return [json.loads(line) for line in raw.splitlines()]

    output['static'] = {method: score(load(f'static-{method}-v2'))
                        for method in ('pixel', 'context')}
    for repeat in ('a', 'b'):
        p, c = (load(f'paced-{method}-{repeat}-v2') for method in ('pixel', 'context'))
        paired_p, paired_c = pair_rows(p, c)
        output['passes'][repeat] = dict(
            all_processed=dict(pixel=score(p), context=score(c)),
            paired_processed=dict(pixel=score(paired_p), context=score(paired_c)),
            scheduling=dict(pixel=scheduling(p), context=scheduling(c)),
            reacquisition=dict(pixel=reacquisition(p), context=reacquisition(c)))
    with args.output.open('x') as stream:
        json.dump(output, stream, indent=2)
        stream.write('\n')
    print(args.output)


if __name__ == '__main__':
    main()
