# OpenLayers Frontend [Beta]

TIAToolbox supports an OpenLayers-based frontend viewer.

There are currently two ways to use the OpenLayers frontend:

- `show-wsi` uses the standard OpenLayers viewer and loads the slide when the
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

## Running the standard viewer

The standard OpenLayers viewer is launched using `show-wsi`.

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

This viewer uses the standard OpenLayers frontend files and keeps the existing
`show-wsi` behaviour.

## Running the experimental dynamic viewer

The experimental OpenLayers viewer is launched using:

```bash
tiatoolbox visualize-beta
```

`visualize-beta` does not require a slide path when the command is started. It
opens an empty viewer and creates a TileServer session which can then be used
to load and change slides dynamically.

The experimental viewer is currently a work in progress. Slides can currently
be loaded from the browser developer console with:

```js
await switchSlide("/path/to/slide.svs");
```

A different slide can be loaded using the same command without restarting the
viewer:

```js
await switchSlide("/path/to/another_slide.svs");
```

The experimental viewer also supports dynamic overlay loading and removal:

```js
await loadOverlay("/path/to/overlay.png");
await removeOverlay("overlay-name");
await clearOverlays();
```

The current slide can be removed and the viewer returned to its empty state
with:

```js
await removeSlide();
```

The same TileServer session remains available, so another slide can be loaded
after removing the current slide.

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

The standard and experimental viewers use separate frontend source files and
Vite build configurations so that they can be developed independently.

The shared frontend files are:

- `package.json` defines the frontend dependencies and build commands.
- `package-lock.json` records the exact dependency versions installed by npm.

The standard viewer files are:

- `src/main.js` contains the OpenLayers viewer used by `show-wsi`.
- `src/style.css` contains the standard viewer styling.
- `vite.config.js` defines how the standard viewer is built.

The experimental viewer files are:

- `src/visualize_beta_viewer.js` contains the dynamic OpenLayers viewer used by
  `visualize-beta`.
- `src/visualize_beta_viewer.css` contains the experimental viewer styling.
- `vite.visualize-beta.config.js` defines how the experimental viewer is built.

The generated files for the standard viewer are:

```text
tiatoolbox/data/visualization/static/openlayers/viewer.js
tiatoolbox/data/visualization/static/openlayers/viewer.css
```

These are served when `show-wsi` is used.

The generated files for the experimental viewer are:

```text
tiatoolbox/data/visualization/static/openlayers/visualize_beta_viewer.js
tiatoolbox/data/visualization/static/openlayers/visualize_beta_viewer.css
```

These are served when `visualize-beta` is used.

The corresponding templates are located at:

```text
tiatoolbox/data/visualization/templates/index.html
tiatoolbox/data/visualization/templates/visualize_beta.html
```

The experimental viewer also uses:

- `tiatoolbox/cli/visualize_beta.py` to provide the `visualize-beta` command.
- `tiatoolbox/visualization/visualize_beta_tileserver.py` to serve the
  experimental viewer template.
- Dynamic TileServer routes in `tiatoolbox/visualization/tileserver.py` for
  loading and removing slides and overlays while the viewer is running.

Run the following frontend development commands from:

```text
tiatoolbox/visualization/openlayers/
```

Install the dependencies with:

```bash
npm ci
```

This installs the exact versions recorded in `package-lock.json`.

### Building the standard viewer

After changing `src/main.js` or `src/style.css`, rebuild the standard viewer
with:

```bash
npm run build
```

This updates:

```text
viewer.js
viewer.css
```

The standard build does not remove the generated experimental viewer files.

### Building the experimental viewer

After changing `src/visualize_beta_viewer.js` or
`src/visualize_beta_viewer.css`, rebuild the experimental viewer with:

```bash
npm run build:visualize-beta
```

This updates:

```text
visualize_beta_viewer.js
visualize_beta_viewer.css
```

The experimental build does not remove the generated standard viewer files.

If changes affect both viewers, both builds can be run:

```bash
npm run build
npm run build:visualize-beta
```

The generated JavaScript and CSS files should be committed together with their
source changes.

Do not edit the generated JavaScript or CSS files directly because they are
generated by Vite.

After rebuilding the frontend, return to the root of the TIAToolbox repository
and test the appropriate viewer.

```bash
cd ../../..
```

Test the standard viewer with:

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
npm run build
npm run build:visualize-beta
```

This updates the recorded dependency version and rebuilds the generated
frontend files using the new version.
