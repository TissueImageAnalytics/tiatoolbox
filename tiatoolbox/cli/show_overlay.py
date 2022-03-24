"""Command line interface for show_overlay"""
from pathlib import Path
import click

from tiatoolbox.cli.common import (
    cli_img_input,
    tiatoolbox_cli,
)


def make_app(slide_path, overlay_path):
    from tiatoolbox.wsicore.wsireader import OpenSlideWSIReader, VirtualWSIReader
    from tiatoolbox.visualization.tileserver import TileServer

    # from flask_cors import CORS
    import argparse
    from PIL import Image
    import numpy as np
    import matplotlib.cm as cm

    wsi = OpenSlideWSIReader(slide_path)
    meta = wsi.info

    if overlay_path.suffix in [".jpg", ".png"]:
        # assume its a low-res heatmap
        im = Image.open(overlay_path)
        if len(im.getbands()) == 1:
            # single channel, make into rgb with colormap
            im = np.array(im)
            c_map = cm.get_cmap("coolwarm")
            im_rgb = (c_map(im) * 255).astype(np.uint8)
            overlay = VirtualWSIReader(im_rgb[:, :, :3], info=meta)
        else:
            overlay = VirtualWSIReader(overlay_path, info=meta)
    else:
        # its a whole slide overlay
        overlay = OpenSlideWSIReader(overlay_path)

    app = TileServer(
        title="Overlay TileServer",
        layers={"slide": wsi, "overlay": overlay},
        add_alpha=True,
        format="webp",
    )

    # CORS(app)
    return app


@tiatoolbox_cli.command()
@cli_img_input()
@click.option("--overlay", help="Path to overlay to be visualised")
def show_overlay(img_input, overlay):
    "show a slide together with a whole slide overlay"
    slide_path = Path(img_input)
    overlay_path = Path(overlay)

    app = make_app(slide_path, overlay_path)

    app.run()
