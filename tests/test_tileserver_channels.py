"""Test tileserver channel configuration."""

from __future__ import annotations

import json

import numpy as np
import pytest

from tiatoolbox.utils.postproc_defs import MultichannelToRGB
from tiatoolbox.visualization.tileserver import TileServer


class _FakeInfo:
    """Simple slide info object carrying mpp, file_path, and slide_dimensions."""

    def __init__(
        self,
        mpp: tuple[float, float] | None = None,
        file_path: str = "slide.svs",
        dims: tuple = (512, 512),
    ) -> None:
        self.mpp = [0.25, 0.25] if mpp is None else mpp
        self.file_path = file_path
        self.slide_dimensions = dims

    def as_dict(self) -> dict:
        return {
            "mpp": self.mpp,
            "file_path": self.file_path,
            "slide_dimensions": self.slide_dimensions,
        }


class _FakeSlide:
    """Slide object that carries .info and .post_proc and records thumbnail calls."""

    def __init__(
        self, info: _FakeInfo = None, post_proc: MultichannelToRGB = None
    ) -> None:
        self.info = info or _FakeInfo()
        self.post_proc = post_proc
        self._thumb_called = 0

    def slide_thumbnail(
        self, *, resolution: float = 8.0, units: str = "mpp"
    ) -> np.ndarray:
        _ = (resolution, units)  # mark as used to satisfy Ruff
        # Parameters are part of the real Slide API; unused by the fake.
        self._thumb_called += 1
        # returning an array avoids PIL issues if used elsewhere
        return np.zeros((8, 8, 3), dtype=np.uint8)


@pytest.fixture
def app() -> TileServer:
    """Create a TileServer app for testing."""
    # Empty app (no default layers); we will create a session first.
    ts = TileServer(title="TS", layers={})
    ts.testing = True
    return ts


@pytest.fixture
def client(app: TileServer) -> TileServer:
    """Create a test client for the TileServer app."""
    return app.test_client()


@pytest.fixture
def session_id(client: TileServer) -> str:
    """Create a session and return its session ID."""
    resp = client.get("/tileserver/session_id")
    cookie = resp.headers.get("Set-Cookie", "")
    # extract cookie value; fallback to "default" if absent
    sid = "default"
    if "session_id=" in cookie:
        sid = cookie.split("session_id=")[1].split(";")[0]
    return sid


def test_get_channels_populated_and_triggers_thumbnail_when_not_validated(
    app: TileServer, client: TileServer, session_id: str
) -> None:
    """Covers lines in get_channels by MultichannelToRGB and is_validated gate."""
    sid = session_id

    # Attach a REAL MultichannelToRGB so isinstance(...) is TRUE
    pp = MultichannelToRGB()
    # Provide some initial channel state and set as 'not validated'
    # Use floats in [0,1], as required by MultichannelToRGB
    pp.color_dict = {"c0": (1.0, 0.0, 0.0), "c1": (0.0, 1.0, 0.0)}
    pp.channels = ["c0", "c1"]
    pp.is_validated = False

    slide = _FakeSlide(info=_FakeInfo(), post_proc=pp)
    app.layers[sid] = {"slide": slide}

    # Call endpoint
    r = client.get("/tileserver/channels")
    assert r.status_code == 200
    payload = r.get_json()
    # Should return the mappings
    # JSON serializes tuples to lists
    assert payload["channels"] == {"c0": [1.0, 0.0, 0.0], "c1": [0.0, 1.0, 0.0]}
    assert payload["active"] == ["c0", "c1"]
    # Should have forced a thumbnail when not validated
    assert slide._thumb_called == 1

    # Call again with already validated state to ensure no extra thumbnailing
    app.layers[sid]["slide"].post_proc.is_validated = True
    r2 = client.get("/tileserver/channels")
    assert r2.status_code == 200
    assert slide._thumb_called == 1  # unchanged


def test_set_channels_updates_dicts_and_marks_unvalidated(
    app: TileServer, client: TileServer, session_id: str
) -> None:
    """Covers set_channels branch."""
    sid = session_id

    pp = MultichannelToRGB()
    pp.color_dict = {"c0": (1.0, 0.0, 0.0)}
    pp.channels = ["c0"]
    pp.is_validated = True  # will be set to False by the route

    slide = _FakeSlide(info=_FakeInfo(), post_proc=pp)
    app.layers[sid] = {"slide": slide}

    # Send lists (JSON friendly) of floats in [0,1]
    new_channels = {"c0": [0.0, 0.0, 1.0], "c2": [0.0, 1.0, 1.0]}
    active = ["c2"]

    r = client.put(
        "/tileserver/channels",
        data={
            "channels": json.dumps(new_channels),
            "active": json.dumps(active),
        },
    )
    assert r.status_code == 200
    assert r.data.decode() == "done"

    # Check mutations
    # Depending on the implementation, color_dict may keep lists or convert to tuples.
    # Accept either representation by normalizing both sides to tuples.
    normalized = {k: tuple(v) for k, v in new_channels.items()}
    assert {k: tuple(v) for k, v in slide.post_proc.color_dict.items()} == normalized
    assert slide.post_proc.channels == active
    assert slide.post_proc.is_validated is False


def test_set_enhance_updates_postproc(
    app: TileServer, client: TileServer, session_id: str
) -> None:
    """Covers set_enhance."""
    sid = session_id

    pp = MultichannelToRGB()
    pp.enhance = 1.0
    slide = _FakeSlide(info=_FakeInfo(), post_proc=pp)
    app.layers[sid] = {"slide": slide}

    r = client.put("/tileserver/enhance", data={"val": json.dumps(1.7)})
    assert r.status_code == 200
    assert r.data.decode() == "done"
    assert slide.post_proc.enhance == 1.7


def test_get_channels_fallback_when_no_multichannel_postproc(
    app: TileServer, client: TileServer, session_id: str
) -> None:
    """Assert fallback path returns empty mappings.

    (Already covered in many suites, but keep it here to be explicit.)
    """
    sid = session_id
    # No post_proc or a different type -> fallback
    app.layers[sid] = {"slide": _FakeSlide(info=_FakeInfo(), post_proc=None)}
    r = client.get("/tileserver/channels")
    assert r.status_code == 200
    assert r.get_json() == {"channels": {}, "active": []}
