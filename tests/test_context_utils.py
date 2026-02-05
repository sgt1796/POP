import pytest

from pop.context import Context
from pop.utils.event_stream import to_event_stream
from pop.utils.json_parse import parse_json, get_value
from pop.utils.validation import validate_not_empty, validate_json
from pop.utils.sanitize_unicode import sanitize
from pop.utils.overflow import truncate_messages
from pop.utils.http_proxy import get_session_with_proxy


def test_context_append_and_to_messages():
    ctx = Context(system="System")
    ctx.append("user", "Hi")
    ctx.append("assistant", "Hello")
    assert ctx.to_messages() == [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello"},
    ]


def test_event_stream_iteration():
    def gen():
        yield {"event": "start"}
        yield {"event": "done"}

    stream = to_event_stream(gen())
    assert list(stream) == [{"event": "start"}, {"event": "done"}]


def test_json_parse_and_get_value():
    data = parse_json('{"a": 1}')
    assert data["a"] == 1
    assert get_value(data, "missing", default="x") == "x"


def test_validation_helpers():
    with pytest.raises(ValueError):
        validate_not_empty("")
    assert validate_json('{"ok": true}') == {"ok": True}
    with pytest.raises(ValueError):
        validate_json("{bad json}")


def test_sanitize_and_truncate():
    assert sanitize("caf\u00e9") == "cafe"
    msgs = ["a" * 3, "b" * 3, "c" * 3]
    assert truncate_messages(msgs, max_length=6) == ["aaa", "bbb"]


def test_get_session_with_proxy():
    session = get_session_with_proxy()
    assert session.trust_env is True
