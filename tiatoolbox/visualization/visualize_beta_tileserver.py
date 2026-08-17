"""Tile server for the experimental OpenLayers viewer."""

from __future__ import annotations

from flask import Response, jsonify
from flask.templating import render_template

from tiatoolbox.visualization.tileserver import TileServer


class VisualizeBetaTileServer(TileServer):
    """TileServer entry point for the experimental OpenLayers viewer."""

    def index(self: VisualizeBetaTileServer) -> str:
        """Serve the experimental OpenLayers viewer."""
        return render_template(
            "visualize_beta.html",
            title=self.title,
            layers="[]",
        )

    def session_id(self: VisualizeBetaTileServer) -> Response:
        """Get or create an experimental viewer session."""
        session_id = self._get_session_id()
        new_session_id = None

        if session_id is None or session_id not in self.layers:
            new_session_id = self._create_session()
            session_id = new_session_id

        response = jsonify({"session_id": session_id})

        if new_session_id is not None:
            response.set_cookie(
                "session_id",
                new_session_id,
                httponly=True,
                samesite="Lax",
            )

        return response
