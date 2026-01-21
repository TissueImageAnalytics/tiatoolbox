"""Tests for channel information retrieval and update in the Bokeh app."""

import json
from types import SimpleNamespace
from typing import Any

from tiatoolbox.visualization.bokeh_app import main


class DummySession:
    """A minimal session mock that records GET and PUT requests."""

    def __init__(self) -> None:
        """Initialize tracking attributes for request inspection."""
        self.last_get: str | None = None
        self.last_put: str | None = None
        self.last_put_data: Any = None

    def get(self, url: str) -> SimpleNamespace:
        """Record GET requests and return a dummy successful response."""
        self.last_get = url
        return SimpleNamespace(
            text=json.dumps({"channels": {"c0": [1, 0, 0]}, "active": []}),
            status_code=200,
        )

    def put(self, url: str, data: str | None = None) -> SimpleNamespace:
        """Record PUT requests and return a dummy successful response."""
        self.last_put = url
        self.last_put_data = data
        return SimpleNamespace(status_code=200)


def test_get_channel_info_uses_configured_port() -> None:
    """Ensure get_channel_info uses the configured host and port."""
    dummy = DummySession()
    # replace UI with a minimal mapping exposing 's'
    old_ui = getattr(main, "UI", None)
    main.UI = {"s": dummy}
    old_host2 = getattr(main, "host2", "127.0.0.1")
    old_port = getattr(main, "port", "5000")
    try:
        main.host2 = "127.0.0.1"
        main.port = "12345"
        channels, _active = main.get_channel_info()
        expected = f"http://{main.host2}:{main.port}/tileserver/channels"
        assert dummy.last_get == expected
        assert "c0" in channels
    finally:
        # restore
        if old_ui is None:
            delattr(main, "UI")
        else:
            main.UI = old_ui
        main.host2 = old_host2
        main.port = old_port


def test_set_channel_info_uses_configured_port() -> None:
    """Ensure set_channel_info uses the configured host and port."""
    dummy = DummySession()
    old_ui = getattr(main, "UI", None)
    main.UI = {"s": dummy}
    old_host2 = getattr(main, "host2", "127.0.0.1")
    old_port = getattr(main, "port", "5000")

    main.host2 = "127.0.0.1"
    main.port = "54321"
    expected = f"http://{main.host2}:{main.port}/tileserver/channels"
    try:
        colors = {"c0": "#ff0000"}
        main.set_channel_info(colors, [0])
    finally:
        assert dummy.last_put == expected
        if old_ui is None:
            delattr(main, "UI")
        else:
            main.UI = old_ui
        main.host2 = old_host2
        main.port = old_port
