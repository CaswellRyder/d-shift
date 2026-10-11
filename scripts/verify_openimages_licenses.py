"""Record current per-photo CC BY links; no dataset admission or legal warranty."""
import argparse
from datetime import datetime, timezone
import hashlib
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import requests

from dtr.data import read_json, sha256, write_json


class LicenseLinks(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        row = dict(attrs)
        if tag == 'a' and 'license' in (row.get('rel') or '').split():
            self.links.append(row.get('href',''))


def cc_by_2(url):
    parsed = urlparse(url)
    return (parsed.scheme == 'https' and parsed.netloc == 'creativecommons.org'
            and parsed.path in ('/licenses/by/2.0/','/licenses/by/2.0/deed.en'))


def verify(root, output):
    root,output = Path(root),Path(output)
    if output.exists():
        raise FileExistsError(output)
    acquisition = read_json(root/'acquisition.json')
    results = []
    for row in acquisition['images']:
        url = row['attribution']['OriginalLandingURL']
        item = dict(image_id=row['image_id'],url=url,verified=False)
        if urlparse(url).netloc != 'www.flickr.com' or not cc_by_2(row['attribution']['License']):
            item['error'] = 'Unexpected source host or license; manual review required'
            results.append(item)
            continue
        try:
            with requests.get(url,stream=True,timeout=(15,20)) as response:
                response.raise_for_status()
                content = bytearray()
                for block in response.iter_content(65536):
                    content.extend(block)
                    if len(content) > 4 << 20:
                        raise ValueError('Photo page exceeds budget')
                parser = LicenseLinks()
                parser.feed(content.decode('utf-8'))
                identity = urlparse(response.url).path.rstrip('/') == urlparse(url).path.rstrip('/')
                item.update(final_url=response.url,page_sha256=hashlib.sha256(content).hexdigest(),
                    license_links=sorted(set(parser.links)),verified=identity and any(cc_by_2(u) for u in parser.links))
        except (requests.RequestException,ValueError,UnicodeError) as error:
            item['error'] = str(error)
        results.append(item)
        print(f"License {len(results)}/{len(acquisition['images'])}: {item['verified']}",flush=True)
    write_json(output,dict(acquisition_sha256=sha256(root/'acquisition.json'),results=results,
        checked_at=datetime.now(timezone.utc).isoformat(),script_sha256=sha256(__file__),
        scope='Source page displays CC BY 2.0 on this date; not an ownership warranty or training approval'))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',required=True)
    p.add_argument('--output',required=True)
    a = p.parse_args()
    verify(a.root,a.output)
