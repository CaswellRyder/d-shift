"""Recheck full replay records before recording ARMv6 compatibility evidence."""
import argparse
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.pi_dense_selection_replay import compare
from scripts.pi_red_blue_bench import verify_bundle


def validate_rows(frames, report, seed):
    if report['seed'] != seed or report['frames'] != len(frames) or len(report['rows']) != len(frames):
        raise ValueError('Incomplete or wrong-seed replay')
    maximum, calls = 0., 0
    for row, actual in zip(frames, report['rows'], strict=True):
        if (actual['id'] != row['id'] or actual['panel'] != row['panel']
                or actual['dense_calls'] != row['dense_calls'] or actual['dense_error'] is not None
                or actual['neural_calls'] != len(row['baseline'])
                or actual['dense_gate'] != bool(row['dense_calls'])):
            raise ValueError('Frame identity, gate or crop-call count changed')
        maximum = max(maximum, compare(actual['baseline'],row['baseline']),
                      compare(actual['detections'],row['selected'][seed]))
        calls += actual['dense_calls']
    if calls != report['dense_calls'] or abs(maximum-report['max_host_score_delta']) > 1e-12:
        raise ValueError('Reported parity totals changed')
    return dict(frame_entries=len(frames), dense_calls=calls, skipped_calls=len(frames)-calls,
                max_host_score_delta=maximum, exact_geometry_and_decisions=True)


def summarize(base, reports):
    base = Path(base)
    bundle = verify_bundle(base)
    frames = read_json(base/'frames.json')
    if len(frames) != bundle['frames']:
        raise ValueError('Incomplete bundle frames')
    sources = {}
    for path, expected in bundle['source_reports'].items():
        if sha256(path) != expected:
            raise ValueError('Development reference changed')
        prior = read_json(path)
        panel = prior['results']['baseline']['frames']
        sources[path] = dict(sha256=expected, frame_entries=len(panel),
            metrics={s:prior['results']['supported'+s]['metrics'] for s in ('47','48')})
    evidence = {}
    for seed, path in reports.items():
        report = read_json(path)
        if (report['machine'] != 'armv6l' or report['runtime'] != 'dtr.native_tflite'
                or report['bundle_sha256'] != sha256(base/'bundle.json')
                or report['classifier_sha256'] != sha256(base/'models/classifier.tflite')
                or report['dense_sha256'] != sha256(base/f'models/dense-{seed}.tflite')
                or report['region_build_receipt']['source_sha256'] != sha256(base/'dtr/native_balloon_regions.c')
                or not report['runtime_library_sha256']
                or report['camera_used'] is not False or report['deployment_approved'] is not False
                or report['isolated_speed_measured'] is not False):
            raise ValueError('Hardware, model or scope identity changed')
        evidence[seed] = dict(validate_rows(frames,report,seed),
            report=str(path), report_sha256=sha256(path),
            classifier_sha256=report['classifier_sha256'], dense_sha256=report['dense_sha256'],
            region_library_sha256=report['region_library_sha256'],
            runtime_library_sha256=report['runtime_library_sha256'],
            health_before=report['health_before'], health_after=report['health_after'])
    if set(evidence) != {'47','48'}:
        raise ValueError('Both fixed seeds are required')
    return dict(models=evidence, bundle_sha256=sha256(base/'bundle.json'),
        development_references=sources,
        unique_processed_image_hashes=len({bundle['files'][r['path']] for r in frames}),
        decision='Both localizer seeds preserve frozen full-pipeline decisions on ARMv6 replay. Remain opt-in research; no active model replacement.',
        caveats=['Repeated, overlapping development panels are not independent qualification.',
                 'Camera app remained running; these are not isolated performance tests.',
                 'Baseline selection is computed additionally for parity diagnostics.',
                 'No new accuracy improvement is established by platform parity.',
                 'Camera capture, temporal tracking and autonomous navigation were not tested.'],
        camera_tested=False, isolated_speed_measured=False, deployment_approved=False,
        qualification_achieved=False)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('base','seed47','seed48','output'):
        p.add_argument('--'+name, type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    result = summarize(a.base, {'47':a.seed47, '48':a.seed48})
    write_json(a.output, result)
    print(result['models'])
