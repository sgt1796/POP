import pytest

from pop.utils.web_snapshot import get_text_snapshot


@pytest.mark.integration
def test_live_web_snapshot(live_enabled):
    if not live_enabled():
        pytest.skip("Live tests disabled.")
    text = get_text_snapshot("https://example.com", use_api_key=False)
    assert isinstance(text, str)
    assert text.strip() != ""
