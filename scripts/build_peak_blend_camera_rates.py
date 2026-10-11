"""Stage the fixed cadence comparison without mutating prior live evidence."""
from pathlib import Path
import shutil

from dtr.data import read_json, sha256, write_json
from scripts.build_peak_blend_live import OUTPUT as SOURCE
from scripts.pi_red_blue_bench import verify_bundle

OUTPUT = Path('runs/peak-blend-camera-rates-pi-20261011')
PROTOCOL = Path('configs/peak-blend-camera-rates-20261011.json')


def build():
    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    source = verify_bundle(SOURCE)
    evidence = Path('research/balloon-red-blue-20261008/peak-blend-live-20261011.json')
    receipt = read_json(evidence)
    if (receipt['bundle_sha256'] != sha256(SOURCE/'bundle.json')
            or receipt['raw']['sha256'] != sha256(receipt['raw']['path'])
            or receipt['host_pi_parity_frames'] != 24):
        raise ValueError('Prior camera evidence changed')
    for relative in source['files']:
        target = OUTPUT/relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(SOURCE/relative, target)
    shutil.copyfile(PROTOCOL, OUTPUT/'camera-rates.json')
    shutil.copyfile('scripts/pi_peak_blend_camera_rates.py', OUTPUT/'scripts/pi_peak_blend_camera_rates.py')
    write_json(OUTPUT/'bundle.json', dict(
        files={str(p.relative_to(OUTPUT)):sha256(p) for p in OUTPUT.rglob('*') if p.is_file()},
        builder_sha256=sha256(__file__), source_bundle_sha256=sha256(SOURCE/'bundle.json'),
        evidence_sha256=sha256(evidence), protocol_sha256=sha256(PROTOCOL),
        deployment_approved=False, qualification_achieved=False))
    print(dict(path=str(OUTPUT), sha256=sha256(OUTPUT/'bundle.json')))


if __name__ == '__main__':
    build()
