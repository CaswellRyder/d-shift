"""TRAIN-only oracle: separate MSER coverage failures from its twelve-crop cutoff.

Truth is used only for diagnosis. No uncapped classifier is executed or proposed
as a deployable policy, and matching ceilings are not achieved detection recall.
"""
from pathlib import Path

import numpy as np

from dtr.data import read_json, sha256, write_json
from dtr.tracking import iou
from dtr.native_balloon_regions import NativeBalloonRegions
from scripts.prepare_source_resolved_negatives import AUDIT, AUDIT_SHA
from scripts.audit_scale_balanced_search import checked_panels
from scripts.scale_balanced_balloon_search import pool
from scripts.research_balloon_search import select
from scripts.audit_balloon_pipeline_errors import maximum_matching

OUTPUT = Path('research/balloon-red-blue-20261008/proposal-ceiling-diagnosis-20261011.json')


def diagnose():
    if OUTPUT.exists():
        raise FileExistsError('Preserve diagnostic receipt')
    if sha256(AUDIT) != AUDIT_SHA:
        raise ValueError('Diagnostic reference changed')
    audit = read_json(AUDIT)
    if sha256(audit['raw']['path']) != audit['raw']['sha256']:
        raise ValueError('Reference predictions changed')
    raw = read_json(audit['raw']['path'])
    native = NativeBalloonRegions('artifacts/balloon-regions-mac-20261008/regions.so', direct=True)
    if read_json(native.path.with_suffix('.json'))['source_sha256'] != sha256('src/dtr/native_balloon_regions.c'):
        raise ValueError('Native source identity changed')
    summary = {}
    for panel, sources in checked_panels().items():
        records = []
        for index, (frame, rgb) in enumerate(sources):
            if 'IMG_' in frame['source'] or not frame['source'].startswith(('train/', 'openimages/train/')):
                raise ValueError('Only admitted TRAIN sources')
            all_regions = pool(rgb, native)
            capped = select(all_regions)
            saved = raw[panel]['corrected47_base_first']['frames'][index]
            if any(saved[k] != frame[k] for k in ('id', 'source', 'truth', 'frame_sha256')):
                raise ValueError('Frame identity changed')
            reference = [r for r in saved['detections'] if r['proposal_origin'] == 'mser']
            for actual, expected in zip(capped, reference, strict=True):
                for key in ('box', 'crop_box', 'proposal_score', 'color_group'):
                    np.testing.assert_allclose(actual[key], expected[key], atol=1e-8, rtol=0)
            variants = {'mser12': capped, 'mser_uncapped_oracle': all_regions}
            for seed in (47, 48):
                variants[f'hybrid{seed}'] = raw[panel][f'corrected{seed}_base_first']['frames'][index]['detections']
            ceilings = {name: {color: len(maximum_matching(
                [t for t in frame['truth'] if t['label'] == color], detections, lambda t, d: True))
                for color in ('red_balloon', 'blue_balloon')} for name, detections in variants.items()}
            misses = []
            for t in frame['truth']:
                best = {name: max((iou(t['box'], r['box']) for r in rows), default=0.)
                        for name, rows in variants.items()}
                if min(best['hybrid47'], best['hybrid48']) >= .5:
                    continue
                nearest = max(all_regions, key=lambda r: iou(t['box'], r['box']), default=None)
                misses.append(dict(target=t, best_iou=best,
                    nearest_mser=nearest, recovered_by_uncapped_geometry=best['mser_uncapped_oracle'] >= .5))
            records.append(dict(id=frame['id'], source=frame['source'], frame_sha256=frame['frame_sha256'],
                truth=frame['truth'], mser_pool_size=len(all_regions), capped_size=len(capped),
                one_to_one_geometry_ceilings=ceilings, hybrid_geometry_misses=misses))
        summary[panel] = dict(frames=records,
            total_mser_regions=sum(r['mser_pool_size'] for r in records),
            maximum_mser_regions=max(r['mser_pool_size'] for r in records),
            one_to_one_geometry_ceilings={name: {c: sum(r['one_to_one_geometry_ceilings'][name][c] for r in records)
                for c in ('red_balloon', 'blue_balloon')} for name in records[0]['one_to_one_geometry_ceilings']},
            hybrid_geometry_misses=sum(len(r['hybrid_geometry_misses']) for r in records),
            misses_covered_by_uncapped_mser=sum(m['recovered_by_uncapped_geometry'] for r in records for m in r['hybrid_geometry_misses']))
        print(panel, {k: v for k, v in summary[panel].items() if k != 'frames'}, flush=True)
    paths = [__file__, AUDIT, 'scripts/prepare_source_resolved_negatives.py',
        'scripts/audit_scale_balanced_search.py', 'scripts/scale_balanced_balloon_search.py',
        'scripts/research_balloon_search.py', 'scripts/research_blue_bright_search.py',
        'scripts/audit_balloon_pipeline_errors.py', 'src/dtr/native_balloon_regions.c', native.path,
        native.path.with_suffix('.json')]
    write_json(OUTPUT, dict(panels=summary, inputs_sha256={str(p): sha256(p) for p in paths},
        source_raw_sha256=audit['raw']['sha256'],
        scope='82TRAIN frames. Native MSER rerun and capped geometry reproduced before an uncapped, truth-dependent geometry oracle. No classifier calls, policy selection, labels, held-out access or runtime modifications. Ceilings do not measure classification accuracy or Pi speed.',
        training_performed=False, development_evaluated=False, test_evaluated=False, pi_tested=False,
        deployment_approved=False, qualification_achieved=False))


if __name__ == '__main__':
    diagnose()
