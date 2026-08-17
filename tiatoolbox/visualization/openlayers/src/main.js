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

const mapElement = document.getElementById("map");

if (mapElement === null) {
  throw new Error("The OpenLayers map element could not be found.");
}

const layersData = JSON.parse(mapElement.dataset.layers ?? "[]");

if (layersData.length === 0) {
  throw new Error("No layers were supplied to the OpenLayers viewer.");
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

const baseSource = layers[0].getSource();
const tileGrid = baseSource.getTileGrid();
const resolutions = tileGrid.getResolutions();
const extent = tileGrid.getExtent();

const projection = new Projection({
  code: "ZoomifyProjection",
  units: "pixels",
  extent,
  metersPerUnit: layersData[0].mpp * 1e-6,
  getPointResolution(resolution) {
    return resolution;
  },
});

// Register the projection for the mouse position and graticule controls.
addProjection(projection);

const view = new View({
  projection,
  resolutions,
  constrainOnlyCenter: true,
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

// Overview map
const overviewMapControl = new OverviewMap({
  className: "ol-overviewmap ol-custom-overviewmap",
  layers: [
    new TileLayer({
      source: baseSource,
    }),
  ],
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

const graticule = new Graticule({
  projection: projection.getCode(),
  margin: graticuleMargin,
  style: graticuleStyle,
  spacing: graticuleSpacing,
  formatCoord(coordinate, position) {
    let displayedCoordinate;

    if (position === "left" || position === "right") {
      displayedCoordinate = -Math.floor(coordinate);
    } else {
      displayedCoordinate = Math.floor(coordinate);
    }

    if (displayedCoordinate >= 1e6) {
      displayedCoordinate = displayedCoordinate.toExponential(3);
      displayedCoordinate = displayedCoordinate.replace("+", "");
    }

    return displayedCoordinate;
  },
});

// Screen-space graticule
const screenSpaceGraticuleSpacing = graticuleSpacing;
const screenSpaceGraticuleMargin = graticuleMargin;

const screenSpaceGraticule = new Graticule({
  projection: projection.getCode(),
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
      let label = "";

      do {
        label += String.fromCharCode(
          65 + (displayedCoordinate % 26),
        );

        displayedCoordinate = Math.floor(
          displayedCoordinate / 26,
        );
      } while (displayedCoordinate > 0);

      return label.split("").reverse().join("");
    }

    return displayedCoordinate;
  },
});

let graticuleToggle;
let screenSpaceGraticuleToggle;

graticuleToggle = new Toggle({
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

screenSpaceGraticuleToggle = new Toggle({
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

map.getView().fit(extent);

// Preserve variables exposed by the original inline viewer.
Object.assign(window, {
  extent,
  fullscreen,
  graticule,
  graticuleToggle,
  layerSwitcher,
  layers,
  layersData,
  map,
  mousePositionControl,
  overviewMapControl,
  projection,
  resolutions,
  rotate,
  scaleLineControl,
  screenSpaceGraticule,
  screenSpaceGraticuleToggle,
  view,
});
