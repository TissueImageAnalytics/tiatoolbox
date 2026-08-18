import "ol/ol.css";
import "ol-ext/dist/ol-ext.css";
import "./style.css";

import FullScreen from "ol/control/FullScreen.js";
import MousePosition from "ol/control/MousePosition.js";
import OverviewMap from "ol/control/OverviewMap.js";
import Rotate from "ol/control/Rotate.js";
import ScaleLine from "ol/control/ScaleLine.js";
import { format as formatCoordinate } from "ol/coordinate.js";
import TileLayer from "ol/layer/Tile.js";
import Map from "ol/Map.js";
import Projection from "ol/proj/Projection.js";
import { addProjection } from "ol/proj.js";
import Zoomify from "ol/source/Zoomify.js";
import Fill from "ol/style/Fill.js";
import Stroke from "ol/style/Stroke.js";
import Style from "ol/style/Style.js";
import Text from "ol/style/Text.js";
import View from "ol/View.js";

import Graticule from "ol-ext/control/Graticule.js";
import LayerSwitcher from "ol-ext/control/LayerSwitcher.js";
import Toggle from "ol-ext/control/Toggle.js";

// Initialise the TileServer session used for dynamic slide loading.
async function createSession() {
  const response = await fetch("/tileserver/session_id");

  if (!response.ok) {
    throw new Error("Failed to create TileServer session.");
  }

  const data = await response.json();

  return data.session_id;
}

// Load a slide into the current TileServer session and return its metadata.
async function loadSlide(slidePath) {
  const formData = new FormData();
  formData.append("slide_path", slidePath);

  const loadResponse = await fetch("/tileserver/slide", {
    method: "PUT",
    body: formData,
  });

  if (!loadResponse.ok) {
    throw new Error(`Failed to load slide: ${slidePath}`);
  }

  const metadataResponse = await fetch("/tileserver/slide");

  if (!metadataResponse.ok) {
    throw new Error("Failed to retrieve slide metadata.");
  }

  return metadataResponse.json();
}

// Create a Zoomify source with versions to avoid reusing tiles from an old slide.
function createSlideSource(sessionId, slideInfo, version) {
  return new Zoomify({
    url:
      `/tileserver/layer/slide/${sessionId}/zoomify/` +
      `{TileGroup}/{z}-{x}-{y}@1x.jpg?v=${version}`,
    size: slideInfo.slide_dimensions,
    crossOrigin: "anonymous",
    zDirection: -1,
  });
}

const mapElement = document.getElementById("map");

if (mapElement === null) {
  throw new Error("The OpenLayers map element could not be found.");
}

let layersData = JSON.parse(mapElement.dataset.layers ?? "[]");
let sessionId = null;
let slideVersion = Date.now();
let overlayVersion = Date.now();
let currentSlideInfo = null;
let currentSlidePath = null;
const overlayLayers = {};

// Dynamic slide loading
const params = new URLSearchParams(window.location.search);
const slidePath = params.get("slide");

if (slidePath !== null) {
  currentSlidePath = slidePath;

  sessionId = await createSession();
  const slideInfo = await loadSlide(slidePath);

  currentSlideInfo = slideInfo;

  layersData = [
    {
      name: "slide",
      url:
        `/tileserver/layer/slide/${sessionId}/zoomify/` +
        `{TileGroup}/{z}-{x}-{y}@1x.jpg?v=${slideVersion}`,
      size: slideInfo.slide_dimensions,
      mpp: slideInfo.mpp[0],
    },
  ];
} else if (layersData.length === 0) {
  sessionId = await createSession();
}

const layers = layersData.map((layer) => {
  const source = new Zoomify({
    url: layer.url,
    size: layer.size,
    crossOrigin: "anonymous",
    zDirection: -1,
  });

  return new TileLayer({
    title: layer.name,
    source,
  });
});

let slideLayer = layers[0];

if (slideLayer === undefined) {
  slideLayer = new TileLayer({
    title: "slide",
  });
  layers.push(slideLayer);
}

const baseSource = slideLayer.getSource();

let resolutions;
let extent;
let projection;

if (baseSource !== null) {
  const tileGrid = baseSource.getTileGrid();

  resolutions = tileGrid.getResolutions();
  extent = tileGrid.getExtent();

  projection = new Projection({
    code: "ZoomifyProjection",
    units: "pixels",
    extent,
    metersPerUnit: layersData[0].mpp * 1e-6,
    getPointResolution(resolution) {
      return resolution;
    },
  });
} else {
  resolutions = [1];
  extent = [0, -1, 1, 0];

  projection = new Projection({
    code: "ZoomifyProjection",
    units: "pixels",
    extent,
    metersPerUnit: 1,
    getPointResolution(resolution) {
      return resolution;
    },
  });
}

// Register the projection for the mouse position and graticule controls.
addProjection(projection);

const view = new View({
  projection,
  resolutions,
  constrainOnlyCenter: true,
  center: [0.5, -0.5],
  resolution: resolutions[0],
});

const map = new Map({
  target: mapElement,
  layers,
  view,
});

// Scale bar
const scaleLineControl = new ScaleLine({
  units: "metric",
  bar: true,
  steps: 10,
  minWidth: 256,
});

map.addControl(scaleLineControl);

const overviewLayer = new TileLayer();

if (baseSource !== null) {
  overviewLayer.setSource(baseSource);
}

// Overview map
const overviewMapControl = new OverviewMap({
  className: "ol-overviewmap ol-custom-overviewmap",
  layers: [overviewLayer],
});

map.addControl(overviewMapControl);

// Mouse position
const coordinateFormat = (coordinate) => {
  const displayedCoordinate = [coordinate[0], -coordinate[1]];

  return formatCoordinate(displayedCoordinate, "{x}, {y}", 0);
};

const mousePositionControl = new MousePosition({
  coordinateFormat,
  projection,
  className: "ol-mouse-position",
  placeholder: "\u00a0",
});

map.addControl(mousePositionControl);

// Rotation reset
const rotate = new Rotate({
  autoHide: false,
  className: "ol-rotate",
});

map.addControl(rotate);

// Fullscreen
const fullscreen = new FullScreen();

map.addControl(fullscreen);

// Layer switcher
const layerSwitcher = new LayerSwitcher();

map.addControl(layerSwitcher);

// Graticule
const graticuleSpacing = 64;
const graticuleMargin = 64;

const graticuleStyle = new Style({
  stroke: new Stroke({
    color: "rgba(0, 0, 0, 0.5)",
    width: 1,
  }),
  text: new Text({
    font: "12px Calibri,sans-serif",
    fill: new Fill({
      color: "rgba(0, 0, 0, 1)",
    }),
    stroke: new Stroke({
      color: "rgba(255, 255, 255, 1)",
      width: 3,
    }),
  }),
});

// Create graticules for the active slide projection.
function createGraticule(graticuleProjection) {
  return new Graticule({
    projection: graticuleProjection,
    margin: graticuleMargin,
    style: graticuleStyle,
    spacing: graticuleSpacing,
    formatCoord: (coordinate, position) => {
      if (position === "left" || position === "right") {
        coordinate = -Math.floor(coordinate);
      } else {
        coordinate = Math.floor(coordinate);
      }

      if (coordinate >= 1e6) {
        coordinate = coordinate.toExponential(3);
        coordinate = coordinate.replace("+", "");
      }

      return coordinate;
    },
  });
}

let graticule = createGraticule(projection);

// Screen-space graticule
const screenSpaceGraticuleSpacing = graticuleSpacing;
const screenSpaceGraticuleMargin = graticuleMargin;

function createScreenSpaceGraticule(graticuleProjection) {
  return new Graticule({
    projection: graticuleProjection.getCode(),
    spacing: screenSpaceGraticuleSpacing,
    margin: screenSpaceGraticuleMargin,
    style: graticuleStyle,
    formatCoord(coordinate, position) {
      const mapExtent = map.getView().calculateExtent(map.getSize());
      const resolution = map.getView().getResolution();

      const xOrigin =
        mapExtent[0] + resolution * screenSpaceGraticuleMargin;
      const yOrigin =
        mapExtent[3] - resolution * screenSpaceGraticuleMargin;

      let displayedCoordinate;

      if (position === "left" || position === "right") {
        displayedCoordinate = -(coordinate - yOrigin);
      } else {
        displayedCoordinate = coordinate - xOrigin;
      }

      displayedCoordinate = Math.floor(
        displayedCoordinate /
          resolution /
          screenSpaceGraticuleSpacing,
      );

      if (position === "left" || position === "right") {
        let string = "";

        do {
          string += String.fromCharCode(
            65 + (displayedCoordinate % 26),
          );
          displayedCoordinate = Math.floor(
            displayedCoordinate / 26,
          );
        } while (displayedCoordinate > 0);

        return string.split("").reverse().join("");
      }

      return displayedCoordinate;
    },
  });
}

let screenSpaceGraticule =
  createScreenSpaceGraticule(projection);

const graticuleToggle = new Toggle({
  html: '<i class="fas fa-ruler-combined"></i>',
  className: "ol-graticule",
  title: "Toggle Graticule",
  onToggle(active) {
    if (active) {
      screenSpaceGraticuleToggle.setActive(false);
      screenSpaceGraticule.setMap(null);
      graticule.setMap(map);
    } else {
      graticule.setMap(null);
    }
  },
});

map.addControl(graticuleToggle);

const screenSpaceGraticuleToggle = new Toggle({
  html: '<i class="fas fa-border-all"></i>',
  className: "ol-screen-space-graticule",
  title: "Toggle Screen Space Graticule",
  onToggle(active) {
    if (active) {
      graticuleToggle.setActive(false);
      graticule.setMap(null);
      screenSpaceGraticule.setMap(map);
    } else {
      screenSpaceGraticule.setMap(null);
    }
  },
});

map.addControl(screenSpaceGraticuleToggle);

if (baseSource !== null) {
  map.getView().fit(extent);
}

const urlViewState = getUrlViewState();

if (urlViewState !== null) {
  map.getView().setCenter(urlViewState.center);
  map.getView().setZoom(urlViewState.zoom);
}

map.on("moveend", () => {
  updateUrlState();
});

function clearOverlayLayers() {
  for (const overlayLayer of Object.values(overlayLayers)) {
    map.removeLayer(overlayLayer);

    const layerIndex = layers.indexOf(overlayLayer);

    if (layerIndex !== -1) {
      layers.splice(layerIndex, 1);
    }
  }

  for (const layerName of Object.keys(overlayLayers)) {
    delete overlayLayers[layerName];
  }
}

async function clearOverlays() {
  const response = await fetch("/tileserver/clear_overlays", {
    method: "PUT",
  });

  if (!response.ok) {
    throw new Error("Failed to clear overlays.");
  }

  clearOverlayLayers();
}

function getUrlViewState() {
  const params = new URLSearchParams(window.location.search);

  const x = Number(params.get("x"));
  const y = Number(params.get("y"));
  const zoom = Number(params.get("zoom"));

  if (
    params.get("x") === null ||
    params.get("y") === null ||
    params.get("zoom") === null ||
    !Number.isFinite(x) ||
    !Number.isFinite(y) ||
    !Number.isFinite(zoom)
  ) {
    return null;
  }

  return {
    center: [x, y],
    zoom,
  };
}

function updateUrlState() {
  if (currentSlidePath === null) {
    return;
  }

  const view = map.getView();
  const center = view.getCenter();
  const zoom = view.getZoom();

  if (
    center === undefined ||
    zoom === undefined
  ) {
    return;
  }

  const url = new URL(window.location.href);

  if (currentSlidePath !== null) {
    url.searchParams.set("slide", currentSlidePath);
  }

  url.searchParams.set("x", center[0].toFixed(2));
  url.searchParams.set("y", center[1].toFixed(2));
  url.searchParams.set("zoom", zoom.toString());

  window.history.replaceState({}, "", url);
}

async function removeSlide() {
  if (sessionId === null) {
    throw new Error("Removing a slide requires a TileServer session.");
  }

  const response = await fetch("/tileserver/slide", {
    method: "DELETE",
  });

  if (!response.ok) {
    throw new Error("Failed to remove the current slide.");
  }

  clearOverlayLayers();

  currentSlideInfo = null;
  currentSlidePath = null;
  layersData.length = 0;

  slideVersion += 1;

  slideLayer.setSource(null);
  overviewLayer.setSource(null);

  const emptyResolutions = [1];
  const emptyExtent = [0, -1, 1, 0];

  const emptyProjection = new Projection({
    code: "ZoomifyProjection",
    units: "pixels",
    extent: emptyExtent,
    metersPerUnit: 1,
    getPointResolution(resolution) {
      return resolution;
    },
  });

  addProjection(emptyProjection);

  const emptyView = new View({
    projection: emptyProjection,
    resolutions: emptyResolutions,
    constrainOnlyCenter: true,
    center: [0.5, -0.5],
    resolution: emptyResolutions[0],
  });

  map.setView(emptyView);

  const overviewMap = overviewMapControl.getOverviewMap();

  overviewMap.setView(
    new View({
      projection: emptyProjection,
      resolutions: emptyResolutions,
      constrainOnlyCenter: true,
      center: [0.5, -0.5],
      resolution: emptyResolutions[0],
    }),
  );

  mousePositionControl.setProjection(emptyProjection);

  graticuleToggle.setActive(false);
  screenSpaceGraticuleToggle.setActive(false);

  graticule.setMap(null);
  screenSpaceGraticule.setMap(null);

  graticule = createGraticule(emptyProjection);
  screenSpaceGraticule =
    createScreenSpaceGraticule(emptyProjection);

  window.graticule = graticule;
  window.screenSpaceGraticule = screenSpaceGraticule;
  window.projection = emptyProjection;
  window.resolutions = emptyResolutions;
  window.extent = emptyExtent;
  window.view = emptyView;

  const url = new URL(window.location.href);
  url.search = "";

  window.history.replaceState({}, "", url);
}

// Slide switching
async function switchSlide(slidePath) {
  if (sessionId === null) {
    throw new Error("Dynamic slide switching requires a TileServer session.");
  }

  const slideInfo = await loadSlide(slidePath);

  currentSlidePath = slidePath;
  clearOverlayLayers();
  currentSlideInfo = slideInfo;

  slideVersion += 1;

  const source = createSlideSource(
    sessionId,
    slideInfo,
    slideVersion,
  );

  const newTileGrid = source.getTileGrid();
  const newExtent = newTileGrid.getExtent();
  const newResolutions = newTileGrid.getResolutions();

  const newProjection = new Projection({
    code: "zoomify",
    units: "pixels",
    extent: newExtent,
    metersPerUnit: slideInfo.mpp[0] * 1e-6,
  });

  addProjection(newProjection);

  // View
  const newCenter = [
    (newExtent[0] + newExtent[2]) / 2,
    (newExtent[1] + newExtent[3]) / 2,
  ];

  const newView = new View({
    projection: newProjection,
    resolutions: newResolutions,
    extent: newExtent,
    constrainOnlyCenter: true,
    center: newCenter,
    resolution: newResolutions[0],
  });

  newView.fit(newExtent, {
    size: map.getSize(),
  });

  map.setView(newView);

  const overviewMap = overviewMapControl.getOverviewMap();

  overviewMap.setView(
    new View({
      projection: newProjection,
      resolutions: newResolutions,
      extent: newExtent,
      constrainOnlyCenter: true,
      center: newCenter,
      resolution: newResolutions[0],
    }),
  );

  const graticuleWasActive = graticuleToggle.getActive();
  const screenSpaceGraticuleWasActive =
    screenSpaceGraticuleToggle.getActive();

  graticule.setMap(null);
  screenSpaceGraticule.setMap(null);

  graticule = createGraticule(newProjection);

  screenSpaceGraticule =
    createScreenSpaceGraticule(newProjection);

  if (graticuleWasActive) {
    graticule.setMap(map);
  }

  if (screenSpaceGraticuleWasActive) {
    screenSpaceGraticule.setMap(map);
  }

  slideLayer.setSource(source);
  overviewLayer.setSource(source);

  window.graticule = graticule;
  window.screenSpaceGraticule = screenSpaceGraticule;

  updateUrlState();
}

async function loadOverlay(overlayPath) {
  if (sessionId === null || currentSlideInfo === null) {
    throw new Error(
      "Dynamic overlay loading requires a loaded slide.",
    );
  }

  const extension = overlayPath
    .split(".")
    .pop()
    .toLowerCase();

  if (extension === "npy" || extension === "mha") {
    throw new Error(
      "Registration overlays are not supported yet.",
    );
  }

  const formData = new FormData();
  formData.append("overlay_path", overlayPath);

  const response = await fetch("/tileserver/overlay", {
    method: "PUT",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Failed to load overlay: ${overlayPath}`);
  }

  const result = await response.json();

  const isAnnotation = ["db", "dat", "geojson"].includes(
    extension,
  );

  const layerName = isAnnotation ? "overlay" : result;

  overlayVersion += 1;

  const source = new Zoomify({
    url:
      `/tileserver/layer/${encodeURIComponent(layerName)}/` +
      `${sessionId}/zoomify/` +
      `{TileGroup}/{z}-{x}-{y}@1x.jpg?v=${overlayVersion}`,
    size: currentSlideInfo.slide_dimensions,
    crossOrigin: "anonymous",
    zDirection: -1,
  });

  if (overlayLayers[layerName] !== undefined) {
    overlayLayers[layerName].setSource(source);
    overlayLayers[layerName].setVisible(true);
  } else {
    const overlayLayer = new TileLayer({
      title: layerName,
      source,
      opacity: 0.75,
    });

    overlayLayers[layerName] = overlayLayer;

    map.addLayer(overlayLayer);
    layers.push(overlayLayer);
  }

  return result;
}

async function removeOverlay(layerName) {
  const overlayLayer = overlayLayers[layerName];

  if (overlayLayer === undefined) {
    throw new Error(`Overlay is not loaded: ${layerName}`);
  }

  const response = await fetch(
    `/tileserver/overlay/${encodeURIComponent(layerName)}`,
    {
      method: "DELETE",
    },
  );

  if (!response.ok) {
    throw new Error(`Failed to remove overlay: ${layerName}`);
  }

  map.removeLayer(overlayLayer);

  const layerIndex = layers.indexOf(overlayLayer);

  if (layerIndex !== -1) {
    layers.splice(layerIndex, 1);
  }

  delete overlayLayers[layerName];
}

async function setAnnotationColors(colorMap) {
  if (overlayLayers.overlay === undefined) {
    throw new Error("No annotation overlay is loaded.");
  }

  const formData = new FormData();
  formData.append(
    "cmap",
    JSON.stringify({
      keys: Object.keys(colorMap),
      values: Object.values(colorMap),
    }),
  );

  const response = await fetch("/tileserver/cmap", {
    method: "PUT",
    body: formData,
  });

  if (!response.ok) {
    throw new Error("Failed to update annotation colours.");
  }

  overlayVersion += 1;

  const source = new Zoomify({
    url:
      `/tileserver/layer/overlay/${sessionId}/zoomify/` +
      `{TileGroup}/{z}-{x}-{y}@1x.jpg?v=${overlayVersion}`,
    size: currentSlideInfo.slide_dimensions,
    crossOrigin: "anonymous",
    zDirection: -1,
  });

  overlayLayers.overlay.setSource(source);
}

// Preserve variables exposed by the original inline viewer.
Object.assign(window, {
  clearOverlays,
  extent,
  fullscreen,
  graticule,
  graticuleToggle,
  layerSwitcher,
  layers,
  layersData,
  loadOverlay,
  map,
  mousePositionControl,
  overlayLayers,
  overviewMapControl,
  projection,
  removeOverlay,
  removeSlide,
  resolutions,
  rotate,
  scaleLineControl,
  screenSpaceGraticule,
  screenSpaceGraticuleToggle,
  setAnnotationColors,
  switchSlide,
  view,
});
