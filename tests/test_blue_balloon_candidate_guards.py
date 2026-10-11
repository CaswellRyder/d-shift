import copy

import pytest

from scripts import blue_balloon_candidate_data as data


def fixture_docs():
    frame = dict(id=0, image_id='source', attribution={'Subset': 'train'}, original_sha256='stamp',
                 original_path='source.jpg', reference_hamming=20, selected_hamming=None)
    review = dict(queue=str(data.QUEUE), queue_sha256='stamp', reviewed_sheets={}, training_approved=True,
                  split='train', deployment_approved=False, exclude=[],
                  labels_by_class={'background': [], 'red_balloon': [], 'blue_balloon': list(range(7))})
    queue = dict(sheets={}, script_sha256='stamp', acquisition_sha256='stamp', license_sha256='stamp',
                 reference_inputs={}, helpers_sha256={}, frames=[frame], samples=[dict(id=i, frame_id=0,
                 split='train', source_image='openimages/train/source') for i in range(7)])
    acquisition = dict(script_sha256='stamp', helpers_sha256={}, selection_sha256='stamp',
                       attribution_sha256='stamp', images=[dict(image_id='source', attribution={'Subset': 'train'},
                                                               sha256='stamp', path='source.jpg')])
    licenses = dict(acquisition_sha256='stamp', script_sha256='stamp', results=[dict(image_id='source', verified=True)])
    return {data.REVIEW: review, data.QUEUE / 'review.json': queue,
            data.ROOT / 'acquisition.json': acquisition, data.ROOT / 'license-verification.json': licenses}


@pytest.mark.parametrize('defect', ['unlicensed', 'wrong_split', 'duplicate', 'identity', 'unapproved', 'unreviewed'])
def test_rejects_invalid_admission_before_reading_pixels(monkeypatch, defect):
    docs = copy.deepcopy(fixture_docs())
    frame = docs[data.QUEUE / 'review.json']['frames'][0]
    if defect == 'unlicensed':
        docs[data.ROOT / 'license-verification.json']['results'][0]['verified'] = False
    elif defect == 'wrong_split':
        frame['attribution']['Subset'] = 'test'
    elif defect == 'duplicate':
        frame['reference_hamming'] = 8
    elif defect == 'identity':
        frame['id'] = 1
    elif defect == 'unapproved':
        docs[data.REVIEW]['training_approved'] = False
    else:
        docs[data.REVIEW]['labels_by_class']['blue_balloon'].pop()
    monkeypatch.setattr(data, 'read_json', lambda path: docs[path])
    monkeypatch.setattr(data, 'sha256', lambda path: 'stamp')
    monkeypatch.setattr(data.Image, 'open', lambda path: pytest.fail('Unsafe source reached pixel read'))
    with pytest.raises(ValueError):
        data.checked_extra()
