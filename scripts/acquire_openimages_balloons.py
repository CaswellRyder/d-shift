"""Bounded TRAIN-only Open Images acquisition; never automatic label admission.

URLs follow the official V7 download page and its CVDF downloader. Metadata is
stream-filtered instead of storing the 638 MB catalog. No validation/test images
are downloaded. Original attribution and annotation flags remain attached.
"""
import argparse
import csv
import hashlib
import io
from pathlib import Path
import re
import time

from PIL import Image
import requests

from dtr.data import sha256, write_json

BOX_URL = 'https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv'
META_URL = 'https://storage.googleapis.com/openimages/2018_04/train/train-images-boxable-with-rotation.csv'
CLASS_URL = 'https://storage.googleapis.com/openimages/v7/oidv7-class-descriptions-boxable.csv'
IMAGE_ROOT = 'https://open-images-dataset.s3.amazonaws.com/train/'
BALLOON = '/m/01j51'


def complete_boxes(raw):
    """Discard partial last line AND its image group at a bounded range edge."""
    text = raw[:raw.rfind(b'\n')+1].decode('utf-8')
    reader = csv.DictReader(io.StringIO(text))
    groups = {}
    last = None
    for row in reader:
        last = row['ImageID']
        if not re.fullmatch('[0-9a-f]{16}', last):
            raise ValueError('Invalid source image ID')
        if row['LabelName'] != BALLOON:
            continue
        box = [float(row[k]) for k in ('XMin','YMin','XMax','YMax')]
        if not (0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1):
            raise ValueError('Invalid source box')
        groups.setdefault(last, []).append(row)
    groups.pop(last, None)
    return groups


def bounded_get(url, limit, headers=None, expected_status=200):
    started = time.monotonic()
    with requests.get(url, headers=headers, stream=True, timeout=(15,30)) as response:
        response.raise_for_status()
        if response.status_code != expected_status:
            raise ValueError('Unexpected response status (range may be ignored)')
        blocks, size = [], 0
        for block in response.iter_content(1 << 20):
            size += len(block)
            if size > limit or time.monotonic()-started > 180:
                raise ValueError('Download budget exceeded')
            blocks.append(block)
        return b''.join(blocks), dict(response.headers)


def metadata_for(image_ids):
    wanted, found, scanned = set(image_ids), {}, 0
    started = time.monotonic()
    with requests.get(META_URL, stream=True, timeout=(15,30)) as response:
        response.raise_for_status()
        if response.status_code != 200:
            raise ValueError('Expected full streaming metadata response')
        headers = dict(response.headers)

        def lines():
            nonlocal scanned
            for line in response.iter_lines(chunk_size=1 << 20):
                scanned += len(line)+1
                if scanned > 650_000_000 or time.monotonic()-started > 600:
                    raise ValueError('Metadata streaming budget exceeded')
                yield line.decode('utf-8')+'\n'

        for n,row in enumerate(csv.DictReader(lines())):
            if row['ImageID'] in wanted:
                if row['Subset'] != 'train' or row['ImageID'] in found:
                    raise ValueError('Unexpected or duplicate metadata entry')
                found[row['ImageID']] = row
            if n % 200000 == 0:
                print(f'Metadata: {n} rows scanned, {len(found)}/{len(wanted)} matched',flush=True)
            if len(found) == len(wanted):
                break
    if set(found) != wanted:
        raise ValueError('Source attribution missing for candidate images')
    return found, dict(headers=headers, approximate_bytes_scanned=scanned)


def acquire(output, scan_bytes=64 << 20, limit=48):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if type(limit) is not int or not 1 <= limit <= 96 or not 1 << 20 <= scan_bytes <= 128 << 20:
        raise ValueError('Require bounded candidate count and annotation scan')
    classes, _ = bounded_get(CLASS_URL, 1 << 20)
    if dict(csv.reader(io.StringIO(classes.decode()))).get(BALLOON) != 'Balloon':
        raise ValueError('Source class identity changed')
    raw, headers = bounded_get(BOX_URL, scan_bytes,
        {'Range':f'bytes=0-{scan_bytes-1}'},206)
    if (len(raw) != scan_bytes or
            not headers.get('Content-Range','').startswith(f'bytes 0-{scan_bytes-1}/')):
        raise ValueError('Incorrect bounded annotation response')
    groups = complete_boxes(raw)
    selected = sorted(groups)[:limit]
    if not selected:
        raise ValueError('No balloon examples in bounded scan')
    print(f'{len(groups)} source TRAIN image groups in range; taking {len(selected)}',flush=True)
    output.mkdir(parents=True)
    write_json(output/'selection.json',dict(annotations_url=BOX_URL,headers=headers,
        scan_bytes=scan_bytes,scan_sha256=hashlib.sha256(raw).hexdigest(),
        available_image_groups=len(groups),selected_image_ids=selected,
        class_csv_sha256=hashlib.sha256(classes).hexdigest(),
        selection='Sorted prefix of bounded TRAIN annotations, not random or independent evaluation',
        boxes={i:groups[i] for i in selected}))
    del raw
    metadata, metadata_receipt = metadata_for(selected)
    write_json(output/'attribution.json',metadata)
    write_json(output/'metadata-source.json',metadata_receipt)
    images, failed = [], []
    (output/'images').mkdir()
    for identifier in selected:
        url = IMAGE_ROOT+identifier+'.jpg'
        try:
            raw, image_headers = bounded_get(url, 12 << 20)
            with Image.open(io.BytesIO(raw)) as im:
                if im.format != 'JPEG' or im.width*im.height > 4_000_000:
                    raise ValueError('Unexpected CVDF image format or dimensions')
                im.load()
                dimensions = list(im.size)
            path = output/'images'/f'{identifier}.jpg'
            path.write_bytes(raw)
            images.append(dict(image_id=identifier,path=f'images/{identifier}.jpg',
                sha256=sha256(path),size=dimensions,url=url,headers=image_headers,
                annotations=groups[identifier],attribution=metadata[identifier]))
            print(f'Image {len(images)}/{len(selected)} {identifier}',flush=True)
        except (requests.RequestException, ValueError, OSError) as error:
            failed.append(dict(image_id=identifier,error=str(error)))
    receipt = dict(source='Open Images TRAIN, CVDF resized originals',images=images,failed=failed,
        selection_sha256=sha256(output/'selection.json'),attribution_sha256=sha256(output/'attribution.json'),
        script_sha256=sha256(__file__),training_approved=False,review_required=True,
        per_image_license_verified=False,duplicate_screened=False,
        source_reported_annotation_license='CC BY 4.0',source_reported_image_license='CC BY 2.0',
        license_caveat='Open Images disclaims per-image license warranties; attribution retained, verify source pages before admission.',
        test_evaluated=False,deployment_approved=False)
    write_json(output/'acquisition.json',receipt)
    print(dict(downloaded=len(images),failed=len(failed),receipt_sha256=sha256(output/'acquisition.json')))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True)
    p.add_argument('--scan-bytes',type=int,default=64 << 20)
    p.add_argument('--limit',type=int,default=48)
    args = p.parse_args()
    acquire(args.output,args.scan_bytes,args.limit)
