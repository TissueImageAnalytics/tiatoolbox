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
const viewerApp = document.querySelector(".viewer-app");

const viewerPanel = document.getElementById("viewer-panel");
const viewerPanelToggle = document.getElementById(
  "viewer-panel-toggle",
);
const currentSlide = document.getElementById("current-slide");

const layerEditor = document.getElementById("layer-editor");
const layerEditorToggle = document.getElementById(
  "layer-editor-toggle",
);
const layerEditorList = document.getElementById(
  "layer-editor-list",
);

if (mapElement === null || viewerApp === null) {
  throw new Error("The OpenLayers viewer could not be found.");
}

if (
  viewerPanel === null ||
  viewerPanelToggle === null ||
  layerEditor === null ||
  layerEditorToggle === null ||
  layerEditorList === null ||
  currentSlide === null
) {
  throw new Error("The OpenLayers viewer controls could not be found.");
}

viewerPanelToggle.addEventListener("click", () => {
  const isHidden = viewerPanel.classList.toggle("hidden");

  viewerPanelToggle.classList.toggle("active", !isHidden);
});

layerEditorToggle.addEventListener("click", () => {
  const isHidden = layerEditor.classList.toggle("hidden");

  layerEditorToggle.classList.toggle("active", !isHidden);
});

let layersData = JSON.parse(mapElement.dataset.layers ?? "[]");
let sessionId = null;
let slideVersion = Date.now();
let overlayVersion = Date.now();
let currentSlideInfo = null;
let currentSlidePath = null;
const overlayLayers = {};

function updateCurrentSlide() {
  if (currentSlidePath === null) {
    currentSlide.textContent = "No slide selected";
    currentSlide.removeAttribute("title");
    return;
  }

  const slideName = currentSlidePath.split(/[\\/]/).pop();

  currentSlide.textContent = slideName || currentSlidePath;
  currentSlide.title = currentSlidePath;
}

// Dynamic slide loading
const params = new URLSearchParams(window.location.search);
const slidePath = params.get("slide");

if (slidePath === null) {
  layersData = [];
  sessionId = await createSession();
} else {
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
}

updateCurrentSlide();

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

slideLayer.setZIndex(0);

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

// Restricting permitted margin around slide
const viewExtentMargin = 0.1;

function getPaddedExtent(slideExtent) {
  const width = slideExtent[2] - slideExtent[0];
  const height = slideExtent[3] - slideExtent[1];

  const xMargin = width * viewExtentMargin;
  const yMargin = height * viewExtentMargin;

  return [
    slideExtent[0] - xMargin,
    slideExtent[1] - yMargin,
    slideExtent[2] + xMargin,
    slideExtent[3] + yMargin,
  ];
}

// Register the projection for the mouse position and graticule controls.
addProjection(projection);

const view = new View({
  projection,
  resolutions,
  extent: getPaddedExtent(extent),
  constrainOnlyCenter: true,
  smoothExtentConstraint: true,
  smoothResolutionConstraint: false,
  center: [0.5, -0.5],
  resolution: resolutions[0],
});

const map = new Map({
  target: mapElement,
  layers,
  view,
});

// Zoom level
const zoomControl = mapElement.querySelector(".ol-zoom");
const zoomOutButton = mapElement.querySelector(".ol-zoom-out");

if (zoomControl === null || zoomOutButton === null) {
  throw new Error("The OpenLayers zoom control could not be found.");
}

const zoomLevel = document.createElement("div");
zoomLevel.className = "ol-zoom-level";

zoomControl.insertBefore(zoomLevel, zoomOutButton);

function updateZoomLevel() {
  const zoom = map.getView().getZoom();

  if (zoom === undefined) {
    return;
  }

  const displayedZoom = Number.isInteger(zoom)
    ? zoom.toString()
    : zoom.toFixed(1);

  zoomLevel.textContent = `${displayedZoom}x`;
}

updateZoomLevel();

// Scalebar
const scaleLineControl = new ScaleLine({
  units: "metric",
  minWidth: 100,
});

map.addControl(scaleLineControl);

const overviewLayer = new TileLayer();

if (baseSource !== null) {
  overviewLayer.setSource(baseSource);
}

const overviewMapWidth = 300;
const overviewMapHeight = 250;

function createOverviewView(overviewProjection, overviewExtent) {
  const center = [
    (overviewExtent[0] + overviewExtent[2]) / 2,
    (overviewExtent[1] + overviewExtent[3]) / 2,
  ];

  const width = overviewExtent[2] - overviewExtent[0];
  const height = overviewExtent[3] - overviewExtent[1];

  const resolution = Math.max(
    width / overviewMapWidth,
    height / overviewMapHeight,
  );

  const overviewView = new View({
    projection: overviewProjection,
    center,
    resolution,
    resolutions: [resolution],
    constrainOnlyCenter: true,
  });

  overviewView.on("change:center", () => {
    const currentCenter = overviewView.getCenter();

    if (
      currentCenter !== undefined &&
      (currentCenter[0] !== center[0] ||
        currentCenter[1] !== center[1])
    ) {
      overviewView.setCenter(center);
    }
  });

  return overviewView;
}

// Overview map
const overviewCollapseLabel = document.createElement("span");
overviewCollapseLabel.className = "overview-toggle-icon";
overviewCollapseLabel.innerHTML =
  '<i class="fas fa-chevron-up"></i>';

const overviewExpandLabel = document.createElement("span");
overviewExpandLabel.className = "overview-toggle-icon";
overviewExpandLabel.innerHTML =
  '<i class="fas fa-chevron-down"></i>';

const overviewMapControl = new OverviewMap({
  className: "ol-overviewmap ol-custom-overviewmap",
  layers: [overviewLayer],
  collapsed: false,
  collapsible: true,
  collapseLabel: overviewCollapseLabel,
  label: overviewExpandLabel,
  rotateWithView: false,
  tipLabel: "Toggle overview map",
  view: createOverviewView(projection, extent),
});

map.addControl(overviewMapControl);

const overviewMap = overviewMapControl.getOverviewMap();

// Mouse position
const coordinateFormat = (coordinate) => {
  const displayedCoordinate = [coordinate[0], -coordinate[1]];

  return formatCoordinate(displayedCoordinate, "{x}, {y}", 0);
};

const mousePositionControl = new MousePosition({
  coordinateFormat,
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
const fullscreen = new FullScreen({
  source: viewerApp,
});

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
    graticuleToggle.element.classList.toggle("active", active);

    if (active) {
      screenSpaceGraticuleToggle.setActive(false);
      screenSpaceGraticuleToggle.element.classList.remove("active");
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
    screenSpaceGraticuleToggle.element.classList.toggle(
      "active",
      active,
    );

    if (active) {
      graticuleToggle.setActive(false);
      graticuleToggle.element.classList.remove("active");
      graticule.setMap(null);
      screenSpaceGraticule.setMap(map);
    } else {
      screenSpaceGraticule.setMap(null);
    }
  },
});

map.addControl(screenSpaceGraticuleToggle);

// Enable or hide controls that require a loaded slide.
function setViewerEnabled(enabled) {
  const zoomInButton = mapElement.querySelector(".ol-zoom-in");
  const zoomOutButton = mapElement.querySelector(".ol-zoom-out");
  const rotateButton = rotate.element.querySelector("button");
  const graticuleButton =
    graticuleToggle.element.querySelector("button");
  const screenSpaceGraticuleButton =
    screenSpaceGraticuleToggle.element.querySelector("button");

  for (const button of [
    zoomInButton,
    zoomOutButton,
    rotateButton,
    graticuleButton,
    screenSpaceGraticuleButton,
  ]) {
    if (button !== null) {
      button.disabled = !enabled;
    }
  }

  scaleLineControl.element.classList.toggle(
    "viewer-control-hidden",
    !enabled,
  );
  mousePositionControl.element.classList.toggle(
    "viewer-control-hidden",
    !enabled,
  );
  overviewMapControl.element.classList.toggle(
    "viewer-control-hidden",
    !enabled,
  );

  if (!enabled) {
    graticuleToggle.setActive(false);
    screenSpaceGraticuleToggle.setActive(false);

    graticuleToggle.element.classList.remove("active");
    screenSpaceGraticuleToggle.element.classList.remove("active");

    graticule.setMap(null);
    screenSpaceGraticule.setMap(null);
  }


  if (enabled) {
    requestAnimationFrame(() => {
      overviewMap.updateSize();
      overviewMap.renderSync();
    });
  }
}

setViewerEnabled(baseSource !== null);

if (baseSource !== null) {
  map.getView().fit(extent);

  const urlViewState = getUrlViewState();

  if (urlViewState !== null) {
    map.getView().setCenter(urlViewState.center);
    map.getView().setZoom(urlViewState.zoom);
  }
}

map.on("moveend", () => {
  updateUrlState();
  updateZoomLevel();
});

function clearOverlayLayers() {
  for (const overlayLayer of Object.values(overlayLayers)) {
    overlayLayer.setSource(null);
    map.removeLayer(overlayLayer);

    const layerIndex = layers.indexOf(overlayLayer);

    if (layerIndex !== -1) {
      layers.splice(layerIndex, 1);
    }
  }

  for (const layerName of Object.keys(overlayLayers)) {
    delete overlayLayers[layerName];
  }

  updateLayerEditor();
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

  url.searchParams.set("slide", currentSlidePath);
  url.searchParams.set("x", center[0].toFixed(2));
  url.searchParams.set("y", center[1].toFixed(2));
  url.searchParams.set("zoom", zoom.toString());

  const search = url.searchParams
    .toString()
    .replace(/%2F/gi, "/");

  window.history.replaceState(
    {},
    "",
    `${url.pathname}?${search}${url.hash}`,
  );
}

async function removeSlide() {
  if (sessionId === null) {
    throw new Error("No TileServer session is available.");
  }

  const response = await fetch("/tileserver/slide", {
    method: "DELETE",
  });

  if (!response.ok) {
    throw new Error("Failed to remove the current slide.");
  }

  clearOverlayLayers();

  currentSlidePath = null;
  currentSlideInfo = null;
  updateCurrentSlide();

  slideVersion += 1;
  overlayVersion += 1;

  slideLayer.setSource(null);
  overviewLayer.setSource(null);

  updateLayerEditor();

  const emptyExtent = [0, -1, 1, 0];
  const emptyResolutions = [1];

  const emptyProjection = new Projection({
    code: "ZoomifyProjectionEmpty",
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

  overviewMap.setView(
    createOverviewView(emptyProjection, emptyExtent),
  );

  graticuleToggle.setActive(false);
  screenSpaceGraticuleToggle.setActive(false);

  graticuleToggle.element.classList.remove("active");
  screenSpaceGraticuleToggle.element.classList.remove("active");

  graticule.setMap(null);
  screenSpaceGraticule.setMap(null);

  graticule = createGraticule(emptyProjection);
  screenSpaceGraticule =
    createScreenSpaceGraticule(emptyProjection);

  window.graticule = graticule;
  window.screenSpaceGraticule = screenSpaceGraticule;

  const url = new URL(window.location.href);
  url.search = "";
  url.hash = "";

  window.history.replaceState({}, "", url);

  setViewerEnabled(false);
  updateZoomLevel();
}

// Slide switching
async function switchSlide(slidePath) {
  if (sessionId === null) {
    throw new Error("Dynamic slide switching requires a TileServer session.");
  }

  clearOverlayLayers();

  const slideInfo = await loadSlide(slidePath);

  currentSlidePath = slidePath;
  updateCurrentSlide();
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
    code: "ZoomifyProjection",
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
    extent: getPaddedExtent(newExtent),
    constrainOnlyCenter: true,
    smoothExtentConstraint: true,
    smoothResolutionConstraint: false,
    center: newCenter,
    resolution: newResolutions[0],
  });

  newView.fit(newExtent, {
    size: map.getSize(),
  });

  map.setView(newView);

  overviewMap.setView(
    createOverviewView(newProjection, newExtent),
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

  updateLayerEditor();

  window.graticule = graticule;
  window.screenSpaceGraticule = screenSpaceGraticule;

  setViewerEnabled(true);
  updateUrlState();
  updateZoomLevel();
}

function getLayerEditorEntries() {
  const entries = [];

  if (slideLayer.getSource() !== null) {
    entries.push({
      name: "slide",
      layer: slideLayer,
    });
  }

  for (const [layerName, layer] of Object.entries(overlayLayers)) {
    entries.push({
      name: layerName,
      layer,
    });
  }

  return entries.sort(
    (a, b) =>
      (b.layer.getZIndex() ?? 0) -
      (a.layer.getZIndex() ?? 0),
  );
}


function moveLayer(layerName, direction) {
  const entries = getLayerEditorEntries();

  const index = entries.findIndex(
    (entry) => entry.name === layerName,
  );

  if (index === -1) {
    return;
  }

  const targetIndex =
    direction === "up" ? index - 1 : index + 1;

  if (
    targetIndex < 0 ||
    targetIndex >= entries.length
  ) {
    return;
  }

  const currentLayer = entries[index].layer;
  const targetLayer = entries[targetIndex].layer;

  const currentZIndex = currentLayer.getZIndex() ?? 0;
  const targetZIndex = targetLayer.getZIndex() ?? 0;

  currentLayer.setZIndex(targetZIndex);
  targetLayer.setZIndex(currentZIndex);

  updateLayerEditor();
}

function updateLayerEditor() {
  layerEditorList.replaceChildren();

  const entries = getLayerEditorEntries();

  if (entries.length === 0) {
    const empty = document.createElement("div");
    empty.className = "layer-editor-empty";
    empty.textContent = "No layers loaded";

    layerEditorList.appendChild(empty);

    return;
  }

  entries.forEach(({ name: layerName, layer }, index) => {
    const item = document.createElement("div");
    item.className = "layer-editor-item";

    const header = document.createElement("div");
    header.className = "layer-editor-item-header";

    const visibility = document.createElement("input");
    visibility.className = "layer-editor-visibility";
    visibility.type = "checkbox";
    visibility.checked = layer.getVisible();
    visibility.title = `Toggle ${layerName}`;

    visibility.addEventListener("change", () => {
      layer.setVisible(visibility.checked);
    });

    const name = document.createElement("span");
    name.className = "layer-editor-name";
    name.textContent = layerName;

    const order = document.createElement("div");
    order.className = "layer-editor-order";

    const moveUp = document.createElement("button");
    moveUp.type = "button";
    moveUp.title = "Move layer up";
    moveUp.innerHTML =
      '<i class="fas fa-chevron-up"></i>';
    moveUp.disabled = index === 0;

    moveUp.addEventListener("click", () => {
      moveLayer(layerName, "up");
    });

    const moveDown = document.createElement("button");
    moveDown.type = "button";
    moveDown.title = "Move layer down";
    moveDown.innerHTML =
      '<i class="fas fa-chevron-down"></i>';
    moveDown.disabled = index === entries.length - 1;

    moveDown.addEventListener("click", () => {
      moveLayer(layerName, "down");
    });

    order.append(moveUp, moveDown);

    header.append(
      visibility,
      name,
      order,
    );

    const opacityRow = document.createElement("div");
    opacityRow.className = "layer-editor-opacity";

    const slider = document.createElement("input");
    slider.className = "layer-editor-slider";
    slider.type = "range";
    slider.min = "0";
    slider.max = "1";
    slider.step = "0.05";
    slider.value = layer.getOpacity().toString();

    const value = document.createElement("span");
    value.className = "layer-editor-value";
    value.textContent =
      `${Math.round(layer.getOpacity() * 100)}%`;

    slider.addEventListener("input", () => {
      const opacity = Number(slider.value);

      layer.setOpacity(opacity);

      value.textContent =
        `${Math.round(opacity * 100)}%`;
    });

    opacityRow.append(slider, value);

    item.append(header, opacityRow);
    layerEditorList.appendChild(item);
  });
}

updateLayerEditor();

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
    const currentLayers = [
      slideLayer,
      ...Object.values(overlayLayers),
    ];

    const highestZIndex = Math.max(
      ...currentLayers.map(
        (layer) => layer.getZIndex() ?? 0,
      ),
    );

    const overlayLayer = new TileLayer({
      title: layerName,
      source,
      opacity: 0.75,
    });

    overlayLayer.setZIndex(highestZIndex + 1);

    overlayLayers[layerName] = overlayLayer;

    map.addLayer(overlayLayer);
    layers.push(overlayLayer);
  }

  updateLayerEditor();

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

  updateLayerEditor();
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
