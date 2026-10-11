import base64
import io

from PIL import Image
import pytest

from scripts.sample_pi_viewer import CONFIG_FIELDS, preview_bytes


def test_preview_validation_and_control_token_not_allowlisted():
    assert 'token' not in CONFIG_FIELDS
    assert preview_bytes({'frame': None}) is None
    buf = io.BytesIO()
    Image.new('RGB', (320, 240)).save(buf, format='JPEG')
    assert preview_bytes({'frame': base64.b64encode(buf.getvalue()).decode()}) == buf.getvalue()
    with pytest.raises(ValueError, match='Viewer reported an error'):
        preview_bytes({'error': 'camera unavailable'})
    buf = io.BytesIO()
    Image.new('RGB', (160, 120)).save(buf, format='JPEG')
    with pytest.raises(ValueError, match='geometry'):
        preview_bytes({'frame': base64.b64encode(buf.getvalue()).decode()})
