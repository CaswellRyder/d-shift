"""Independently recount joint TRAIN decisions and locate incumbent-target losses."""
from pathlib import Path

from dtr.data import read_json, sha256, write_json
from scripts.audit_joint_balanced_balloon import OUTPUT as AUDIT, PANELS, paired_gate
from scripts.audit_balanced_localizer import screen
from scripts.audit_balloon_pipeline_errors import analyze_frame
from scripts.diagnose_clutter_stage_attribution import difference

OUTPUT = Path('research/balloon-red-blue-20261008/joint-balanced-balloon-decision-20261011.json')


def summarize():
    if OUTPUT.exists():
        raise FileExistsError('Preserve decision receipt')
    report = read_json(AUDIT)
    for path, digest in {**report['inputs_sha256'], **report['models_sha256']}.items():
        if sha256(path) != digest:
            raise ValueError('Composition inputs changed')
    if sha256(report['raw']['path']) != report['raw']['sha256']:
        raise ValueError('Saved detections changed')
    raw = read_json(report['raw']['path'])
    if set(raw) != set(PANELS):
        raise ValueError('Panel coverage changed')
    recounted, losses = {}, {}
    for panel, variants in raw.items():
        recounted[panel], losses[panel] = {}, {}
        baseline = variants['parent_parent47_current']
        expected_names = {f'{loc}_{cla}{seed}_{policy}'
                          for loc in ('parent', 'control', 'balanced')
                          for cla in ('parent', 'control', 'admit')
                          for seed in (47, 48) for policy in ('current', 'reserved')}
        if set(variants) != expected_names:
            raise ValueError('Factorial controls incomplete')
        for name, result in variants.items():
            if len(result['frames']) != PANELS[panel] or any(
                    f['neural_calls'] != len(f['detections']) or f['neural_calls'] > 12
                    for f in result['frames']):
                raise ValueError('Coverage or inference budget changed')
            fresh = screen(result, baseline)
            expected = {k: v for k, v in report['panels'][panel][name].items() if k != 'comparisons'}
            if fresh != expected:
                raise ValueError('Recounted scene metrics or retention disagree')
            recounted[panel][name] = fresh
            if not name.startswith('balanced_admit'):
                continue
            suffix = name.removeprefix('balanced_admit')
            comparisons = {
                'full_matched_control': difference(result, variants[f'control_control{suffix}']),
                'localizer_only': difference(result, variants[f'balanced_parent{suffix}']),
                'classifier_only': difference(result, variants[f'parent_admit{suffix}'])}
            if comparisons != report['panels'][panel][name]['comparisons']:
                raise ValueError('Component effect does not reproduce')
            rows = []
            for lost in fresh['effect']['lost_targets']:
                frame = next(f for f in result['frames'] if f['source'] == lost['source'])
                target_index = frame['truth'].index(lost['target'])
                reasons = [e for e in analyze_frame(frame)['false_negatives'] if e['index'] == target_index]
                if len(reasons) != 1:
                    raise ValueError('Lost target is not exactly one false negative')
                rows.append(dict(lost, diagnosis=reasons[0]))
            losses[panel][name] = rows
    gates = {policy: paired_gate(recounted, policy) for policy in ('current', 'reserved')}
    if gates != report['eligible_for_separate_export_development']:
        raise ValueError('Eligibility decision changed')
    write_json(OUTPUT, dict(audit_sha256=sha256(AUDIT), raw_sha256=report['raw']['sha256'],
        script_sha256=sha256(__file__), eligible_for_separate_export_development=gates,
        lost_target_diagnosis=losses, decision=report['decision'],
        next_experiment='Investigate proposal provenance and suppression: several old targets have localized boxes but are suppressed after new learned boxes are added. Freeze a source-aware merge experiment before replay; separately address classifier boundary regressions. Do not tune on IMG_ development or reserved qualification scenes.',
        scope='Independent recount of all36factorial variants across82TRAIN scenes. Error attribution is truth-assisted diagnosis, not an inference policy or held-out result. Components and composition remain unpromoted.',
        training_performed=False, development_evaluated=False, test_evaluated=False,
        pi_tested=False, deployment_approved=False, qualification_achieved=False))
    print(dict(eligible=gates, all_variants_recounted=36, frames_per_variant=82), flush=True)


if __name__ == '__main__':
    summarize()
