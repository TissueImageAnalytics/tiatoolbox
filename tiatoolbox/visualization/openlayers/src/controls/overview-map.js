import OverviewMap from "ol/control/OverviewMap.js";
import TileLayer from "ol/layer/Tile.js";
import View from "ol/View.js";

const overviewMapSizes = {
  small: {
    width: 220,
    height: 180,
  },
  default: {
    width: 300,
    height: 250,
  },
  large: {
    width: 380,
    height: 320,
  },
};

function createOverviewMapController({
  map,
  source,
  projection,
  extent,
  sizeSelect,
  visibleInput,
  hasSlide,
}) {
  const overviewLayer = new TileLayer();

  if (source !== null) {
    overviewLayer.setSource(source);
  }

  function getSize() {
    return (
      overviewMapSizes[sizeSelect.value] ??
      overviewMapSizes.default
    );
  }

  function createView(
    overviewProjection,
    overviewExtent,
  ) {
    const overviewMapSize = getSize();

    const center = [
      (overviewExtent[0] + overviewExtent[2]) / 2,
      (overviewExtent[1] + overviewExtent[3]) / 2,
    ];

    const width =
      overviewExtent[2] - overviewExtent[0];

    const height =
      overviewExtent[3] - overviewExtent[1];

    const resolution = Math.max(
      width / overviewMapSize.width,
      height / overviewMapSize.height,
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

  const collapseLabel = document.createElement("span");
  collapseLabel.className = "overview-toggle-icon";
  collapseLabel.innerHTML =
    '<i class="fas fa-chevron-up"></i>';

  const expandLabel = document.createElement("span");
  expandLabel.className = "overview-toggle-icon";
  expandLabel.innerHTML =
    '<i class="fas fa-chevron-down"></i>';

  const control = new OverviewMap({
    className: "ol-overviewmap ol-custom-overviewmap",
    layers: [overviewLayer],
    collapsed: false,
    collapsible: true,
    collapseLabel,
    label: expandLabel,
    rotateWithView: false,
    tipLabel: "Toggle overview map",
    view: createView(projection, extent),
  });

  map.addControl(control);

  const overviewMap = control.getOverviewMap();

  function refresh() {
    requestAnimationFrame(() => {
      overviewMap.updateSize();
      overviewMap.renderSync();
    });
  }

  function updateSize() {
    const size = getSize();

    control.element.style.setProperty(
      "--overview-map-width",
      `${size.width}px`,
    );

    control.element.style.setProperty(
      "--overview-map-height",
      `${size.height}px`,
    );

    overviewMap.updateSize();

    const currentSource = overviewLayer.getSource();

    if (currentSource !== null) {
      const currentProjection =
        map.getView().getProjection();

      const currentExtent =
        currentSource.getTileGrid().getExtent();

      overviewMap.setView(
        createView(
          currentProjection,
          currentExtent,
        ),
      );
    }

    overviewMap.renderSync();
  }

  function updateVisibility() {
    const visible =
      hasSlide() && visibleInput.checked;

    control.element.classList.toggle(
      "viewer-control-hidden",
      !visible,
    );

    if (visible) {
      refresh();
    }
  }

  function setViewerEnabled(enabled) {
    control.element.classList.toggle(
      "viewer-control-hidden",
      !enabled || !visibleInput.checked,
    );

    if (enabled) {
      refresh();
    }
  }

  function setSource(newSource) {
    overviewLayer.setSource(newSource);
  }

  function setView(
    overviewProjection,
    overviewExtent,
  ) {
    overviewMap.setView(
      createView(
        overviewProjection,
        overviewExtent,
      ),
    );
  }

  sizeSelect.addEventListener("change", () => {
    updateSize();
  });

  updateSize();

  return {
    control,
    refresh,
    setSource,
    setView,
    setViewerEnabled,
    updateSize,
    updateVisibility,
  };
}

export { createOverviewMapController };
