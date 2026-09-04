import FullScreen from "ol/control/FullScreen.js";
import MousePosition from "ol/control/MousePosition.js";
import Rotate from "ol/control/Rotate.js";
import Zoom from "ol/control/Zoom.js";
import { format as formatCoordinate } from "ol/coordinate.js";
import MouseWheelZoom from "ol/interaction/MouseWheelZoom.js";

function createMapControlsController({
  map,
  viewerApp,
  getSlideSource,
  zoomVisibleInput,
  zoomLevelVisibleInput,
  rotationVisibleInput,
  resetViewButton,
  resetViewControl,
  resetViewVisibleInput,
  fullscreenVisibleInput,
  mousePositionVisibleInput,
  mouseWheelZoomSensitivitySelect,
  zoomButtonStepSelect,
}) {
  const mouseWheelZoomPresets = {
    low: {
      deltaPerZoom: 600,
      maxDelta: 1,
    },

    default: {
      deltaPerZoom: 300,
      maxDelta: 1,
    },

    high: {
      deltaPerZoom: 150,
      maxDelta: 2,
    },
  };

  function createMouseWheelZoomInteraction() {
    const preset =
      mouseWheelZoomPresets[
        mouseWheelZoomSensitivitySelect.value
      ] ?? mouseWheelZoomPresets.default;

    const interaction = new MouseWheelZoom({
      maxDelta: preset.maxDelta,
    });

    // OpenLayers does not expose mouse-wheel sensitivity publicly.
    // deltaPerZoom_ is private API, so verify this when upgrading OpenLayers.
    interaction.deltaPerZoom_ =
      preset.deltaPerZoom;

    interaction.setActive(
      getSlideSource() !== null,
    );

    return interaction;
  }

  let mouseWheelZoomInteraction =
    createMouseWheelZoomInteraction();

  map.addInteraction(mouseWheelZoomInteraction);

  function createZoomControl() {
    return new Zoom({
      delta: Number(zoomButtonStepSelect.value),
    });
  }

  let zoomControlInstance =
    createZoomControl();

  map.addControl(zoomControlInstance);

  // Zoom level
  let zoomControl = zoomControlInstance.element;

  let zoomOutButton =
    zoomControl.querySelector(".ol-zoom-out");

  if (zoomOutButton === null) {
    throw new Error(
      "The OpenLayers zoom control could not be found.",
    );
  }

  const zoomLevel =
    document.createElement("input");

  zoomLevel.type = "number";
  zoomLevel.className = "ol-zoom-level";
  zoomLevel.step = "1";
  zoomLevel.setAttribute(
    "aria-label",
    "Zoom level",
  );
  zoomLevel.title = "Zoom level";

  zoomControl.insertBefore(
    zoomLevel,
    zoomOutButton,
  );

  function updateZoomLevel() {
    const zoom = map.getView().getZoom();

    if (zoom === undefined) {
      zoomLevel.value = "";
      return;
    }

    zoomLevel.value = Number.isInteger(zoom)
      ? zoom.toString()
      : zoom.toFixed(1);
  }

  function applyZoomLevel() {
    const zoom =
      Number.parseFloat(zoomLevel.value);

    if (!Number.isFinite(zoom)) {
      updateZoomLevel();
      return;
    }

    const view = map.getView();

    const clampedZoom = Math.min(
      Math.max(zoom, view.getMinZoom()),
      view.getMaxZoom(),
    );

    view.setZoom(clampedZoom);
    updateZoomLevel();
  }

  zoomLevel.addEventListener("focus", () => {
    zoomLevel.select();
  });

  zoomLevel.addEventListener("blur", () => {
    applyZoomLevel();
  });

  zoomLevel.addEventListener(
    "keydown",
    (event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        zoomLevel.blur();
        return;
      }

      if (event.key === "Escape") {
        event.preventDefault();
        updateZoomLevel();
        zoomLevel.blur();
      }
    },
  );

  updateZoomLevel();

  function resetView() {
    const source = getSlideSource();

    if (source === null) {
      return;
    }

    const slideExtent = source
      .getTileGrid()
      .getExtent();

    const currentView = map.getView();

    currentView.setRotation(0);

    currentView.fit(slideExtent, {
      size: map.getSize(),
    });
  }

  resetViewButton.addEventListener(
    "click",
    () => {
      resetView();
    },
  );

  // Mouse position
  const coordinateFormat = (coordinate) => {
    const displayedCoordinate = [
      coordinate[0],
      -coordinate[1],
    ];

    return formatCoordinate(
      displayedCoordinate,
      "{x}, {y}",
      0,
    );
  };

  const mousePositionControl =
    new MousePosition({
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

  const bottomControlsGroup =
    document.createElement("div");

  bottomControlsGroup.className =
    "bottom-controls-group ol-unselectable";

  viewerApp.append(bottomControlsGroup);

  bottomControlsGroup.append(
    resetViewControl,
    zoomControl,
    fullscreen.element,
  );

  let viewerEnabled =
    getSlideSource() !== null;

  function setViewerEnabled(enabled) {
    viewerEnabled = enabled;

    const zoomInButton =
      zoomControl.querySelector(".ol-zoom-in");

    const zoomOutButton =
      zoomControl.querySelector(".ol-zoom-out");

    const rotateButton =
      rotate.element.querySelector("button");

    for (const button of [
      zoomInButton,
      zoomOutButton,
      rotateButton,
      resetViewButton,
    ]) {
      if (button !== null) {
        button.disabled = !enabled;
      }
    }

    zoomLevel.disabled = !enabled;

    zoomControl.classList.toggle(
      "viewer-control-disabled",
      !enabled,
    );

    mouseWheelZoomInteraction.setActive(enabled);

    mousePositionControl.element.classList.toggle(
      "viewer-control-hidden",
      !enabled ||
        !mousePositionVisibleInput.checked,
    );
  }

  function updateVisibility() {
    const hasSlide =
      getSlideSource() !== null;

    zoomControl.classList.toggle(
      "viewer-control-hidden",
      !zoomVisibleInput.checked,
    );

    zoomLevel.classList.toggle(
      "viewer-control-hidden",
      !zoomLevelVisibleInput.checked,
    );

    rotate.element.classList.toggle(
      "viewer-control-hidden",
      !rotationVisibleInput.checked,
    );

    resetViewControl.classList.toggle(
      "viewer-control-hidden",
      !resetViewVisibleInput.checked,
    );

    fullscreen.element.classList.toggle(
      "viewer-control-hidden",
      !fullscreenVisibleInput.checked,
    );

    mousePositionControl.element.classList.toggle(
      "viewer-control-hidden",
      !hasSlide ||
        !mousePositionVisibleInput.checked,
    );
  }

  function updateMouseWheelZoomSensitivity() {
    map.removeInteraction(
      mouseWheelZoomInteraction,
    );

    mouseWheelZoomInteraction =
      createMouseWheelZoomInteraction();

    mouseWheelZoomInteraction.setActive(
      viewerEnabled,
    );

    map.addInteraction(
      mouseWheelZoomInteraction,
    );
  }

  function updateZoomButtonStep() {
    map.removeControl(zoomControlInstance);

    zoomControlInstance =
      createZoomControl();

    map.addControl(zoomControlInstance);

    zoomControl = zoomControlInstance.element;

    zoomOutButton =
      zoomControl.querySelector(".ol-zoom-out");

    if (zoomOutButton === null) {
      throw new Error(
        "The OpenLayers zoom control could not be found.",
      );
    }

    zoomControl.insertBefore(
      zoomLevel,
      zoomOutButton,
    );

    bottomControlsGroup.insertBefore(
      zoomControl,
      fullscreen.element,
    );

    setViewerEnabled(viewerEnabled);
    updateVisibility();
  }

  mouseWheelZoomSensitivitySelect.addEventListener(
    "change",
    () => {
      updateMouseWheelZoomSensitivity();
    },
  );

  zoomButtonStepSelect.addEventListener(
    "change",
    () => {
      updateZoomButtonStep();
    },
  );

  return {
    fullscreen,
    mousePositionControl,
    rotate,
    setViewerEnabled,
    updateMouseWheelZoomSensitivity,
    updateVisibility,
    updateZoomButtonStep,
    updateZoomLevel,
  };
}

export { createMapControlsController };
