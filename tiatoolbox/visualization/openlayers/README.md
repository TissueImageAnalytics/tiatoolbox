# OpenLayers Frontend [Beta]

TIAToolbox supports an OpenLayers-based frontend viewer.

There are currently two ways to use the OpenLayers frontend:

- `show-wsi` uses the legacy OpenLayers viewer and loads the slide when the
  command is started.
- `visualize-beta` launches the experimental dynamic OpenLayers viewer, where
  slides and overlays can be loaded or changed after the viewer has started.

The experimental viewer is being developed separately so that new viewer
functionality can be tested without changing the existing `show-wsi` workflow.

## Normal users

Normal TIAToolbox users do not need Node.js, npm, Vite or `node_modules`
to run either viewer.

The generated JavaScript and CSS files are included in the Python package, so
TIAToolbox can be installed and both OpenLayers viewers can be used directly
through the `show-wsi` and `visualize-beta` commands.

The npm/Vite setup described below is only required when developing or
rebuilding the OpenLayers frontend.

## Running the legacy viewer

The legacy OpenLayers viewer is launched using `show-wsi`.

Run `show-wsi` with the path to a whole slide image:

```bash
tiatoolbox show-wsi --img-input /path/to/slide.svs
```

Multiple image paths can also be provided to display a slide together with
additional layers. Each path is provided using a separate `--img-input`
option:

```bash
tiatoolbox show-wsi \
  --img-input /path/to/slide.svs \
  --img-input /path/to/overlay.png
```

Names can also be assigned to the layers using `--name`. If names are
provided, the number of names must match the number of image paths:

```bash
tiatoolbox show-wsi \
  --img-input /path/to/slide.svs \
  --img-input /path/to/tissue-mask.png \
  --img-input /path/to/semantic-segmentation.db \
  --name slide \
  --name tissue-mask \
  --name semantic-segmentation
```

Annotation layers can also be coloured using `--colour-by` together with
`--colour-map`. For example:

```bash
tiatoolbox show-wsi \
  --img-input /path/to/slide.svs \
  --img-input /path/to/tissue-mask.png \
  --img-input /path/to/semantic-segmentation.db \
  --img-input /path/to/nucleus-detections.db \
  --name slide \
  --name tissue-mask \
  --name semantic-segmentation \
  --name nucleus-detections \
  --colour-by type \
  --colour-map categorical
```

The slide and any additional layers are loaded when the server starts.

This viewer uses the legacy OpenLayers frontend files and keeps the existing
`show-wsi` behaviour.

## Running the experimental dynamic viewer

The experimental OpenLayers viewer is launched using:

```bash
tiatoolbox visualize-beta
```

`visualize-beta` does not require a slide path when the command is started. It
opens an empty viewer and creates a TileServer session which can then be used
to load and change slides dynamically.

Slides and overlays can be loaded directly from the Files panel in the top-left
corner of the viewer.

Use **Load Slide** to open the native file picker and select a whole slide image.
A different slide can be selected at any time without restarting the viewer.

Once a slide is loaded, use **Load Overlay** to select an image or annotation
overlay. Multiple overlays can be loaded and managed using the Layers panel.

The slide and overlay file pickers remember their most recently used directories
separately for the current TileServer session. For example, selecting a slide
from a slides directory does not change the directory used when opening the
overlay file picker.

Use **Clear Overlays** to remove all overlays while keeping the current slide
loaded. Use **Clear Slide** to remove the slide and its overlays and return the
viewer to its empty state.

The native file picker currently supports local Windows and WSL environments.

The experimental viewer also stores the current slide, position and zoom level
in the URL. A saved viewer URL can therefore be reopened to restore the slide
and view state.

These features are currently part of the experimental `visualize-beta`
workflow and do not change the behaviour of `show-wsi`.

## For Developers

Node.js and npm are required when changing or rebuilding the OpenLayers frontend.

Check that they are installed with:

```bash
node --version
npm --version
```

The existing `show-wsi` viewer is kept separately as the legacy viewer, while
the experimental dynamic viewer uses the main OpenLayers frontend files.

The shared frontend files are:

- `package.json` defines the frontend dependencies and build commands.
- `package-lock.json` records the exact dependency versions installed by npm.

### Legacy viewer

The legacy viewer files are:

- `src/main_legacy.js` contains the OpenLayers viewer used by `show-wsi`.
- `src/style_legacy.css` contains the legacy viewer styling.
- `vite.legacy.config.js` defines how the legacy viewer is built.

The generated files for the legacy viewer are:

```text
tiatoolbox/data/visualization/static/openlayers/viewer_legacy.js
tiatoolbox/data/visualization/static/openlayers/viewer_legacy.css
```

These are served when `show-wsi` is used.

The legacy viewer template is:

```text
tiatoolbox/data/visualization/templates/index_legacy.html
```

### Experimental viewer

The experimental viewer files are:

- `src/main.js` contains the dynamic OpenLayers viewer used by
  `visualize-beta`.
- `src/style.css` contains the experimental viewer styling.
- `vite.config.js` defines how the experimental viewer is built.

The generated files for the experimental viewer are:

```text
tiatoolbox/data/visualization/static/openlayers/viewer.js
tiatoolbox/data/visualization/static/openlayers/viewer.css
```

These are served when `visualize-beta` is used.

The experimental viewer template is:

```text
tiatoolbox/data/visualization/templates/index.html
```

### TileServer

Both viewers share the same TileServer implementation in:

```text
tiatoolbox/visualization/tileserver.py
```

`TileServer` uses its `legacy` option to select the appropriate frontend and
session behaviour.

`show-wsi` starts the TileServer with:

```python
legacy = True
```

`visualize-beta` starts the TileServer with:

```python
legacy = False
```

The experimental viewer also uses:

- `tiatoolbox/cli/visualize_beta.py` to provide the `visualize-beta` command.
- Dynamic TileServer routes in `tiatoolbox/visualization/tileserver.py` for
  selecting, loading and removing slides and overlays while the viewer is
  running.

## Building the frontend

Run the frontend development commands from:

```text
tiatoolbox/visualization/openlayers/
```

Install the dependencies with:

```bash
npm ci
```

This installs the exact versions recorded in `package-lock.json`.

### Building the legacy viewer

After changing `src/main_legacy.js` or `src/style_legacy.css`, rebuild the
legacy viewer with:

```bash
npm run build:legacy
```

This updates:

```text
viewer_legacy.js
viewer_legacy.css
```

The legacy build does not remove the generated experimental viewer files.

### Building the experimental viewer

After changing `src/main.js` or `src/style.css`, rebuild the experimental
viewer with:

```bash
npm run build
```

This updates:

```text
viewer.js
viewer.css
```

The experimental build does not remove the generated legacy viewer files.

If changes affect both viewers, both builds can be run:

```bash
npm run build:legacy
npm run build
```

The generated JavaScript and CSS files should be committed together with their
source changes.

Do not edit the generated JavaScript or CSS files directly because they are
generated by Vite.

After rebuilding the frontend, return to the root of the TIAToolbox repository:

```bash
cd ../../..
```

Test the legacy viewer with:

```bash
tiatoolbox show-wsi --img-input /path/to/slide.svs
```

Test the experimental viewer with:

```bash
tiatoolbox visualize-beta
```

If the OpenLayers or ol-ext version needs to be changed, update it with npm
and then rebuild both viewers as required. For example:

```bash
cd tiatoolbox/visualization/openlayers/
npm install --save-exact ol@<version>
npm run build:legacy
npm run build
```

This updates the recorded dependency version and rebuilds the generated
frontend files using the new version.
