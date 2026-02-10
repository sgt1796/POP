import POP.utils.web_snapshot as ws


def test_web_snapshot_headers_and_url(monkeypatch):
    captured = {}

    class DummyResponse:
        text = "ok"

        def raise_for_status(self):
            return None

    def dummy_get(url, headers=None):
        captured["url"] = url
        captured["headers"] = headers or {}
        return DummyResponse()

    monkeypatch.setattr(ws.requests, "get", dummy_get)

    ws.get_text_snapshot(
        "https://example.com",
        use_api_key=False,
        target_selector=["main", ".content"],
        wait_for_selector=["#ready"],
        exclude_selector=[".ads"],
        links_at_end=True,
        images_at_end=True,
        image_caption=True,
    )

    assert captured["url"] == "https://r.jina.ai/https://example.com"
    headers = captured["headers"]
    assert headers["X-Target-Selector"] == "main,.content"
    assert headers["X-Wait-For-Selector"] == "#ready"
    assert headers["X-Remove-Selector"] == ".ads"
    assert headers["X-With-Links-Summary"] == "true"
    assert headers["X-With-Images-Summary"] == "true"
    assert headers["X-With-Generated-Alt"] == "true"


def test_web_snapshot_request_exception(monkeypatch):
    def dummy_get(url, headers=None):
        raise ws.requests.exceptions.RequestException("boom")

    monkeypatch.setattr(ws.requests, "get", dummy_get)
    result = ws.get_text_snapshot("https://example.com", use_api_key=False)
    assert "Error fetching text snapshot" in result
