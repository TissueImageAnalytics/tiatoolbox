# OpenLayers JavaScript tests

The JavaScript tests for the OpenLayers frontend use Vitest with jsdom.

Run the following commands from:

```text
tiatoolbox/visualization/openlayers/
```

Run all JavaScript tests with:

```bash
npm run test
```

Run the tests with coverage using:

```bash
npm run test:coverage
```

To rerun tests automatically while developing:

```bash
npm run test:watch
```

A specific test file can also be run directly. For example:

```bash
npm run test -- ../../../tests/javascript/panels/layers.test.js
```

Run the JavaScript linter with:

```bash
npm run lint
```

Tests that exercise rendering or interaction with the TileServer should also be
checked manually using the appropriate OpenLayers viewer after rebuilding the
frontend.
