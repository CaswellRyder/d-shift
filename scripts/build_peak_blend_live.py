"""Immutable live research bundle, preserving the previously tested replay bundle."""
from pathlib import Path
import shutil

from dtr.data import read_json, sha256, write_json
from scripts.pi_red_blue_bench import verify_bundle

SOURCE = Path('runs/blue-peak-blend-pi-20261011')
OUTPUT = Path('runs/peak-blend-live-pi-20261011')


def build():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = verify_bundle(SOURCE)
    decision_path = Path('research/balloon-red-blue-20261008/blue-peak-blend-decision-20261011.json')
    decision = read_json(decision_path)
    if (decision['bundle_sha256'] != sha256(SOURCE/'bundle.json')
            or decision['pi_report']['sha256'] != sha256(decision['pi_report']['path'])
            or decision['policies']['head_blend']['timed_frames'] != 282):
        raise ValueError('Paired Pi evidence changed')
    for relative in source['files']:
        target = OUTPUT/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE/relative, target)
    for name in ('pi_peak_blend_live', 'pi_balloon_live_bench'):
        shutil.copyfile('scripts/'+name+'.py', OUTPUT/'scripts'/f'{name}.py')
    write_json(OUTPUT/'bundle.json', dict(
        files={str(p.relative_to(OUTPUT)):sha256(p) for p in OUTPUT.rglob('*') if p.is_file()},
        source_bundle_sha256=sha256(SOURCE/'bundle.json'), decision_sha256=sha256(decision_path),
        builder_sha256=sha256(__file__), protocol=dict(policy='head_blend', count=24, seconds=30, fps=5),
        scope='Bounded single-pipeline unlabeled camera smoke; no active-model replacement or actuation.',
        deployment_approved=False, qualification_achieved=False))
    print(dict(path=str(OUTPUT), sha256=sha256(OUTPUT/'bundle.json')))


if __name__ == '__main__':
    build()
