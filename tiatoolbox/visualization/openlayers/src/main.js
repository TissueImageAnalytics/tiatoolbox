import "ol/ol.css";
import "ol-ext/dist/ol-ext.css";
import "./style.css";

import FullScreen from "ol/control/FullScreen.js";
import Zoom from "ol/control/Zoom.js";
import { defaults as defaultControls } from "ol/control/defaults.js";
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

import MouseWheelZoom from "ol/interaction/MouseWheelZoom.js";
import { defaults as defaultInteractions } from "ol/interaction/defaults.js";

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

// Get files from a directory configured when TileServer was launched.
async function getConfiguredFiles(kind) {
  const response = await fetch(`/tileserver/files/${kind}`);

  if (!response.ok) {
    throw new Error(`Failed to get configured ${kind} files.`);
  }

  return response.json();
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
const viewerFiles = document.getElementById("viewer-files");

const layerEditor = document.getElementById("layer-editor");
const layerEditorToggle = document.getElementById(
  "layer-editor-toggle",
);

const layerEditorList = document.getElementById(
  "layer-editor-list",
);

const settingsPanel = document.getElementById(
  "settings-panel",
);

const settingsToggle = document.getElementById(
  "settings-toggle",
);

const settingsCloseButton = document.getElementById(
  "settings-close",
);

const settingsTabs = document.querySelectorAll(
  ".settings-tab",
);

const settingsTabPanels = document.querySelectorAll(
  ".settings-tab-panel",
);

const zoomVisibleInput = document.getElementById(
  "settings-zoom-visible",
);

const zoomLevelVisibleInput = document.getElementById(
  "settings-zoom-level-visible",
);

const rotationVisibleInput = document.getElementById(
  "settings-rotation-visible",
);

const graticuleVisibleInput = document.getElementById(
  "settings-graticule-visible",
);

const screenSpaceGraticuleVisibleInput =
  document.getElementById(
    "settings-screen-space-graticule-visible",
  );

const resetViewButton = document.getElementById(
  "reset-view-button",
);

const resetViewControl = document.querySelector(
  ".reset-view-control",
);

const resetViewVisibleInput = document.getElementById(
  "settings-reset-view-visible",
);

const fullscreenVisibleInput = document.getElementById(
  "settings-fullscreen-visible",
);

const mousePositionVisibleInput = document.getElementById(
  "settings-mouse-position-visible",
);

const overviewMapVisibleInput = document.getElementById(
  "settings-overview-map-visible",
);

const overviewMapSizeSelect = document.getElementById(
  "settings-overview-map-size",
);

const mouseWheelZoomSensitivitySelect =
  document.getElementById(
    "settings-mouse-wheel-zoom-sensitivity",
  );

const zoomButtonStepSelect =
  document.getElementById(
    "settings-zoom-button-step",
  );

const scaleBarEnabledInput = document.getElementById(
  "settings-scale-bar-enabled",
);

const themeSelect = document.getElementById(
  "settings-theme",
);

const gridThemeSelect = document.getElementById(
  "settings-grid-theme",
);

const gridOpacityInput = document.getElementById(
  "settings-grid-opacity",
);

const gridOpacityValue = document.getElementById(
  "settings-grid-opacity-value",
);

const gridSpacingSelect = document.getElementById(
  "settings-grid-spacing",
);

const gridLabelsVisibleInput = document.getElementById(
  "settings-grid-labels-visible",
);

const controlOpacityInput = document.getElementById(
  "settings-control-opacity",
);

const controlOpacityValue = document.getElementById(
  "settings-control-opacity-value",
);

const resetDefaultsButton = document.getElementById(
  "settings-reset-defaults",
);

const scaleBarColourInput = document.getElementById(
  "settings-scale-bar-colour",
);

const scaleBarOpacityInput = document.getElementById(
  "settings-scale-bar-opacity",
);

const scaleBarOpacityValue = document.getElementById(
  "settings-scale-bar-opacity-value",
);

const scaleBarSizeSelect = document.getElementById(
  "settings-scale-bar-size",
);

const scaleBarUnitsSelect = document.getElementById(
  "settings-scale-bar-units",
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
  settingsPanel === null ||
  settingsToggle === null ||
  settingsCloseButton === null ||
  zoomVisibleInput === null ||
  zoomLevelVisibleInput === null ||
  rotationVisibleInput === null ||
  graticuleVisibleInput === null ||
  screenSpaceGraticuleVisibleInput === null ||
  resetViewButton === null ||
  resetViewControl === null ||
  resetViewVisibleInput === null ||
  fullscreenVisibleInput === null ||
  mousePositionVisibleInput === null ||
  overviewMapVisibleInput === null ||
  overviewMapSizeSelect === null ||
  mouseWheelZoomSensitivitySelect === null ||
  zoomButtonStepSelect === null ||
  themeSelect === null ||
  gridThemeSelect === null ||
  gridOpacityInput === null ||
  gridOpacityValue === null ||
  gridSpacingSelect === null ||
  gridLabelsVisibleInput === null ||
  controlOpacityInput === null ||
  controlOpacityValue === null ||
  resetDefaultsButton === null ||
  scaleBarEnabledInput === null ||
  scaleBarColourInput === null ||
  scaleBarOpacityInput === null ||
  scaleBarOpacityValue === null ||
  scaleBarSizeSelect === null ||
  scaleBarUnitsSelect === null ||
  viewerFiles === null
) {
  throw new Error("The OpenLayers viewer controls could not be found.");
}

const settingsStorageKey =
  "tiatoolbox-openlayers-settings";

function saveSettings() {
  const settings = {
    theme: themeSelect.value,
    interfaceOpacity: controlOpacityInput.value,

    controls: {
      zoom: zoomVisibleInput.checked,
      zoomLevel: zoomLevelVisibleInput.checked,
      rotation: rotationVisibleInput.checked,
      graticule: graticuleVisibleInput.checked,
      screenSpaceGraticule:
        screenSpaceGraticuleVisibleInput.checked,
      resetView: resetViewVisibleInput.checked,
      fullscreen: fullscreenVisibleInput.checked,
      mousePosition:
        mousePositionVisibleInput.checked,
      overviewMap: overviewMapVisibleInput.checked,
    },

    navigation: {
      mouseWheelZoomSensitivity:
        mouseWheelZoomSensitivitySelect.value,
      zoomButtonStep:
        zoomButtonStepSelect.value,
    },

    overviewMap: {
      size: overviewMapSizeSelect.value,
    },

    grid: {
      theme: gridThemeSelect.value,
      opacity: gridOpacityInput.value,
      spacing: gridSpacingSelect.value,
      labels: gridLabelsVisibleInput.checked,
    },

    scaleBar: {
      enabled: scaleBarEnabledInput.checked,
      colour: scaleBarColourInput.value,
      opacity: scaleBarOpacityInput.value,
      size: scaleBarSizeSelect.value,
      units: scaleBarUnitsSelect.value,
    },
  };

  try {
    window.localStorage.setItem(
      settingsStorageKey,
      JSON.stringify(settings),
    );
  } catch {
    // Continue using the viewer if storage is unavailable.
  }
}

function loadSettings() {
  let savedSettings;

  try {
    const storedSettings =
      window.localStorage.getItem(settingsStorageKey);

    if (storedSettings === null) {
      return;
    }

    savedSettings = JSON.parse(storedSettings);
  } catch {
    return;
  }

  if (
    savedSettings === null ||
    typeof savedSettings !== "object"
  ) {
    return;
  }

  if (
    ["dark", "light", "high-contrast"].includes(
      savedSettings.theme,
    )
  ) {
    themeSelect.value = savedSettings.theme;
  }

  if (
    ["40", "45", "50", "55", "60", "65", "70",
      "75", "80", "85", "90", "95", "100"].includes(
      savedSettings.interfaceOpacity,
    )
  ) {
    controlOpacityInput.value =
      savedSettings.interfaceOpacity;
  }

  const controls = savedSettings.controls;

  if (
    controls !== null &&
    typeof controls === "object"
  ) {
    if (typeof controls.zoom === "boolean") {
      zoomVisibleInput.checked = controls.zoom;
    }

    if (typeof controls.zoomLevel === "boolean") {
      zoomLevelVisibleInput.checked =
        controls.zoomLevel;
    }

    if (typeof controls.rotation === "boolean") {
      rotationVisibleInput.checked =
        controls.rotation;
    }

    if (typeof controls.graticule === "boolean") {
      graticuleVisibleInput.checked =
        controls.graticule;
    }

    if (
      typeof controls.screenSpaceGraticule ===
      "boolean"
    ) {
      screenSpaceGraticuleVisibleInput.checked =
        controls.screenSpaceGraticule;
    }

    if (typeof controls.resetView === "boolean") {
      resetViewVisibleInput.checked =
        controls.resetView;
    }

    if (typeof controls.fullscreen === "boolean") {
      fullscreenVisibleInput.checked =
        controls.fullscreen;
    }

    if (
      typeof controls.mousePosition === "boolean"
    ) {
      mousePositionVisibleInput.checked =
        controls.mousePosition;
    }

    if (
      typeof controls.overviewMap === "boolean"
    ) {
      overviewMapVisibleInput.checked =
        controls.overviewMap;
    }
  }

  const navigation = savedSettings.navigation;

  if (
    navigation !== null &&
    typeof navigation === "object"
  ) {
    if (
      ["low", "default", "high"].includes(
        navigation.mouseWheelZoomSensitivity,
      )
    ) {
      mouseWheelZoomSensitivitySelect.value =
        navigation.mouseWheelZoomSensitivity;
    }
    if (
      ["0.1", "0.5", "1", "2"].includes(
        navigation.zoomButtonStep,
      )
    ) {
      zoomButtonStepSelect.value =
        navigation.zoomButtonStep;
    }
  }

  const overviewMapSettings = savedSettings.overviewMap;

  if (
    overviewMapSettings !== null &&
    typeof overviewMapSettings === "object"
  ) {
    if (
      ["small", "default", "large"].includes(
        overviewMapSettings.size,
      )
    ) {
      overviewMapSizeSelect.value =
        overviewMapSettings.size;
    }
  }

  const grid = savedSettings.grid;

  if (
    grid !== null &&
    typeof grid === "object"
  ) {
    if (
      [
        "default",
        "light",
        "dark",
        "light-contrast",
        "dark-contrast",
      ].includes(grid.theme)
    ) {
      gridThemeSelect.value = grid.theme;
    }

    const opacity = Number(grid.opacity);

    if (
      Number.isFinite(opacity) &&
      opacity >= 0 &&
      opacity <= 100
    ) {
      gridOpacityInput.value = opacity.toString();
    }

    if (
      ["fine", "default", "coarse"].includes(
        grid.spacing,
      )
    ) {
      gridSpacingSelect.value = grid.spacing;
    }

    if (typeof grid.labels === "boolean") {
      gridLabelsVisibleInput.checked = grid.labels;
    }
  }

  const scaleBar = savedSettings.scaleBar;

  if (
    scaleBar !== null &&
    typeof scaleBar === "object"
  ) {
    if (typeof scaleBar.enabled === "boolean") {
      scaleBarEnabledInput.checked =
        scaleBar.enabled;
    }

    if (
      typeof scaleBar.colour === "string" &&
      /^#[0-9a-fA-F]{6}$/.test(scaleBar.colour)
    ) {
      scaleBarColourInput.value = scaleBar.colour;
    }

    const opacity = Number(scaleBar.opacity);

    if (
      Number.isFinite(opacity) &&
      opacity >= 0 &&
      opacity <= 100
    ) {
      scaleBarOpacityInput.value =
        opacity.toString();
    }

    if (
      ["small", "default", "large"].includes(
        scaleBar.size,
      )
    ) {
      scaleBarSizeSelect.value = scaleBar.size;
    }

    if (
      ["metric", "imperial"].includes(
        scaleBar.units,
      )
    ) {
      scaleBarUnitsSelect.value = scaleBar.units;
    }
  }
}

function hexToRgb(hex) {
  const value = hex.replace("#", "");

  if (!/^[0-9a-fA-F]{6}$/.test(value)) {
    return null;
  }

  return {
    r: Number.parseInt(value.slice(0, 2), 16),
    g: Number.parseInt(value.slice(2, 4), 16),
    b: Number.parseInt(value.slice(4, 6), 16),
  };
}

function getRelativeLuminance({ r, g, b }) {
  const channels = [r, g, b].map((channel) => {
    const value = channel / 255;

    return value <= 0.04045
      ? value / 12.92
      : ((value + 0.055) / 1.055) ** 2.4;
  });

  return (
    0.2126 * channels[0] +
    0.7152 * channels[1] +
    0.0722 * channels[2]
  );
}

function getContrastingColour(rgb) {
  const luminance = getRelativeLuminance(rgb);

  const whiteContrast = 1.05 / (luminance + 0.05);
  const blackContrast = (luminance + 0.05) / 0.05;

  return whiteContrast >= blackContrast
    ? "#ffffff"
    : "#000000";
}

function mixColour(rgb, target, amount) {
  return {
    r: Math.round(rgb.r + (target - rgb.r) * amount),
    g: Math.round(rgb.g + (target - rgb.g) * amount),
    b: Math.round(rgb.b + (target - rgb.b) * amount),
  };
}

function toRgba(rgb, opacity) {
  return `rgba(${rgb.r}, ${rgb.g}, ${rgb.b}, ${opacity})`;
}

const interfaceThemeColours = {
  dark: "#111111",
  light: "#f2f2f2",
  "high-contrast": "#000000",
};

const scaleBarThemeColours = {
  dark: "#ffffff",
  light: "#000000",
  "high-contrast": "#ffffff",
};

function updateControlAppearance() {
  const themeColour =
    interfaceThemeColours[themeSelect.value] ??
    interfaceThemeColours.dark;

  const colour = hexToRgb(themeColour);

  if (colour === null) {
    return;
  }

  const opacity =
    Number(controlOpacityInput.value) / 100;

  const foreground = getContrastingColour(colour);

  const foregroundRgb =
    foreground === "#ffffff"
      ? { r: 255, g: 255, b: 255 }
      : { r: 0, g: 0, b: 0 };

  const shadowColour =
    foreground === "#ffffff"
      ? { r: 0, g: 0, b: 0 }
      : { r: 255, g: 255, b: 255 };

  const interactionTarget =
    foreground === "#ffffff" ? 255 : 0;

  const surfaceColour = mixColour(
    colour,
    interactionTarget,
    0.08,
  );

  const hoverColour = mixColour(
    colour,
    interactionTarget,
    0.16,
  );

  const pressedColour = mixColour(
    colour,
    interactionTarget,
    0.28,
  );

  const borderColour = mixColour(
    colour,
    interactionTarget,
    0.4,
  );

  const focusBorderColour = mixColour(
    colour,
    interactionTarget,
    0.58,
  );

  viewerApp.style.setProperty(
    "--viewer-control-background",
    toRgba(colour, opacity),
  );

  viewerApp.style.setProperty(
    "--viewer-control-surface-background",
    toRgba(surfaceColour, opacity),
  );

  viewerApp.style.setProperty(
    "--viewer-control-hover-background",
    toRgba(hoverColour, opacity),
  );

  viewerApp.style.setProperty(
    "--viewer-control-pressed-background",
    toRgba(pressedColour, opacity),
  );

  viewerApp.style.setProperty(
    "--viewer-control-foreground",
    foreground,
  );

  viewerApp.style.setProperty(
    "--viewer-control-hover-foreground",
    getContrastingColour(hoverColour),
  );

  viewerApp.style.setProperty(
    "--viewer-control-pressed-foreground",
    getContrastingColour(pressedColour),
  );

  viewerApp.style.setProperty(
    "--viewer-control-muted-foreground",
    toRgba(foregroundRgb, 0.7),
  );

  viewerApp.style.setProperty(
    "--viewer-control-subtle-foreground",
    toRgba(foregroundRgb, 0.55),
  );

  viewerApp.style.setProperty(
    "--viewer-control-border",
    toRgba(
      borderColour,
      Math.max(opacity, 0.7),
    ),
  );

  viewerApp.style.setProperty(
    "--viewer-control-focus-border",
    toRgba(
      focusBorderColour,
      Math.max(opacity, 0.9),
    ),
  );

  viewerApp.style.setProperty(
    "--viewer-control-foreground-shadow",
    toRgba(shadowColour, 0.75),
  );

  controlOpacityValue.textContent =
    `${controlOpacityInput.value}%`;
}

themeSelect.addEventListener("change", () => {
  updateControlAppearance();

  scaleBarColourInput.value =
    scaleBarThemeColours[themeSelect.value] ??
    scaleBarThemeColours.dark;

  updateScaleBarColour();
  updateScaleBarOpacity();
  updateGridAppearance();
});

controlOpacityInput.addEventListener("input", () => {
  updateControlAppearance();
});

loadSettings();

updateControlAppearance();

let scaleBarEnabled = scaleBarEnabledInput.checked;

function setViewerPanelOpen(open) {
  if (open) {
    setLayerEditorOpen(false);
  }

  viewerPanel.classList.toggle("hidden", !open);
  viewerPanelToggle.classList.toggle("active", open);
  viewerPanelToggle.innerHTML = open
    ? '<i class="fas fa-folder-open"></i>'
    : '<i class="fas fa-folder"></i>';

  if (!open) {
    for (const select of viewerPanel.querySelectorAll(
      ".viewer-file-select.open",
    )) {
      select.close?.();
    }
  }
}

function setLayerEditorOpen(open) {
  if (open) {
    setViewerPanelOpen(false);
  }

  layerEditor.classList.toggle("hidden", !open);
  layerEditorToggle.classList.toggle("active", open);
}

function setSettingsPanelOpen(open) {
  settingsPanel.classList.toggle("hidden", !open);
  settingsToggle.classList.toggle("active", open);
}

setViewerPanelOpen(true);

viewerPanelToggle.addEventListener("click", () => {
  const open = viewerPanel.classList.contains("hidden");
  setViewerPanelOpen(open);
});

layerEditorToggle.addEventListener("click", () => {
  const open = layerEditor.classList.contains("hidden");
  setLayerEditorOpen(open);
});

settingsToggle.addEventListener("click", () => {
  const open = settingsPanel.classList.contains("hidden");
  setSettingsPanelOpen(open);
});

settingsCloseButton.addEventListener("click", () => {
  setSettingsPanelOpen(false);
});

for (const tab of settingsTabs) {
  tab.addEventListener("click", () => {
    const selectedTab = tab.dataset.settingsTab;

    for (const otherTab of settingsTabs) {
      otherTab.classList.toggle(
        "active",
        otherTab === tab,
      );
    }

    for (const panel of settingsTabPanels) {
      panel.classList.toggle(
        "hidden",
        panel.dataset.settingsPanel !== selectedTab,
      );
    }
  });
}

let layersData = JSON.parse(mapElement.dataset.layers ?? "[]");
let sessionId = null;
let slideVersion = Date.now();
let overlayVersion = Date.now();
let currentSlideInfo = null;
let currentSlidePath = null;
const overlayLayers = {};
const annotationLayerNames = new Set();

const fileSelectors = document.createElement("div");
fileSelectors.className = "viewer-file-selectors";

let fileSelectId = 0;

function createFileSelect(placeholder) {
  const select = document.createElement("div");
  select.className = "viewer-file-select";

  const button = document.createElement("button");
  button.type = "button";
  button.className = "viewer-file-select-button";
  button.setAttribute("aria-haspopup", "listbox");
  button.setAttribute("aria-expanded", "false");

  const label = document.createElement("span");
  label.className = "viewer-file-select-label";
  label.textContent = placeholder;

  button.append(label);

  const menu = document.createElement("div");
  menu.className = "viewer-file-select-menu";
  menu.hidden = true;

  const search = document.createElement("input");
  search.type = "text";
  search.className = "viewer-file-select-search";
  search.placeholder = "Search";
  search.autocomplete = "off";
  search.spellcheck = false;
  search.setAttribute(
    "aria-label",
    `Search ${placeholder.toLowerCase()}`,
  );

  const options = document.createElement("div");
  options.className = "viewer-file-select-options";
  options.id = `viewer-file-select-${fileSelectId}`;
  options.setAttribute("role", "listbox");

  fileSelectId += 1;

  button.setAttribute("aria-controls", options.id);
  search.setAttribute("aria-controls", options.id);

  menu.append(search, options);
  select.append(button, menu);

  let files = [];
  let selectedPath = "";
  let currentPlaceholder = placeholder;
  let filteredFiles = [];
  let activeIndex = -1;

  function setExpanded(expanded) {
    select.classList.toggle("open", expanded);
    menu.hidden = !expanded;
    button.setAttribute(
      "aria-expanded",
      expanded.toString(),
    );
  }

  function updateLabel() {
    if (selectedPath === "") {
      label.textContent = currentPlaceholder;
      label.title = "";
      return;
    }

    const selectedFile = files.find(
      (file) => file.path === selectedPath,
    );

    const fileName =
      selectedFile?.name ??
      selectedPath.split(/[\\/]/).pop() ??
      selectedPath;

    label.textContent = fileName;
    label.title = selectedPath;
  }

  function selectFile(file) {
    selectedPath = file.path;
    updateLabel();
    close();

    select.dispatchEvent(
      new CustomEvent("change", {
        detail: file.path,
      }),
    );
  }

  function renderOptions() {
    const query = search.value
      .trim()
      .toLocaleLowerCase();

    filteredFiles = files.filter(
      (file) =>
        file.name
          .toLocaleLowerCase()
          .includes(query),
    );

    options.replaceChildren();

    if (filteredFiles.length === 0) {
      const empty = document.createElement("div");
      empty.className = "viewer-file-select-empty";
      empty.textContent = "No matches";
      options.append(empty);
      return;
    }

    filteredFiles.forEach((file, index) => {
      const option = document.createElement("button");

      option.type = "button";
      option.className =
        "viewer-file-select-option";
      option.textContent = file.name;
      option.title = file.path;
      option.setAttribute("role", "option");
      option.setAttribute(
        "aria-selected",
        (file.path === selectedPath).toString(),
      );

      if (file.path === selectedPath) {
        option.classList.add("selected");
      }

      if (index === activeIndex) {
        option.classList.add("active");
      }

      option.addEventListener("mousedown", (event) => {
        event.preventDefault();
      });

      option.addEventListener("click", (event) => {
        event.stopPropagation();
        selectFile(file);
      });

      options.append(option);
    });

    options
      .querySelector(
        ".viewer-file-select-option.active",
      )
      ?.scrollIntoView({
        block: "nearest",
      });
  }

  function open() {
    if (button.disabled || files.length === 0) {
      return;
    }

    for (const otherSelect of document.querySelectorAll(
      ".viewer-file-select.open",
    )) {
      if (otherSelect !== select) {
        otherSelect.close?.();
      }
    }

    search.value = "";
    activeIndex = -1;
    renderOptions();
    setExpanded(true);

    requestAnimationFrame(() => {
      search.focus();
    });
  }

  function close() {
    search.value = "";
    activeIndex = -1;
    setExpanded(false);
  }

  select.close = close;

  select.setFiles = (
    newFiles,
    newPlaceholder,
  ) => {
    files = newFiles;
    selectedPath = "";
    currentPlaceholder = newPlaceholder;

    updateLabel();
    close();

    select.disabled = files.length === 0;
  };

  Object.defineProperty(select, "value", {
    get() {
      return selectedPath;
    },
    set(filePath) {
      selectedPath = filePath;
      updateLabel();
      close();
    },
  });

  Object.defineProperty(select, "disabled", {
    get() {
      return button.disabled;
    },
    set(disabled) {
      button.disabled = disabled;

      select.classList.toggle(
        "disabled",
        disabled,
      );

      if (disabled) {
        close();
      }
    },
  });

  button.addEventListener("click", () => {
    if (select.classList.contains("open")) {
      close();
      return;
    }

    open();
  });

  button.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      open();
    }
  });

  search.addEventListener("input", () => {
    activeIndex = -1;
    renderOptions();
  });

  search.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      close();
      button.focus();
      return;
    }

    if (filteredFiles.length === 0) {
      return;
    }

    if (event.key === "ArrowDown") {
      event.preventDefault();

      activeIndex = Math.min(
        activeIndex + 1,
        filteredFiles.length - 1,
      );

      renderOptions();
      return;
    }

    if (event.key === "ArrowUp") {
      event.preventDefault();

      activeIndex =
        activeIndex <= 0
          ? filteredFiles.length - 1
          : activeIndex - 1;

      renderOptions();
      return;
    }

    if (event.key === "Enter") {
      event.preventDefault();

      const index =
        activeIndex >= 0 ? activeIndex : 0;

      const file = filteredFiles[index];

      if (file !== undefined) {
        selectFile(file);
      }
    }
  });

  document.addEventListener("click", (event) => {
    if (!select.contains(event.target)) {
      close();
    }
  });

  select.disabled = true;

  return select;
}

const slideSelect = createFileSelect("Select slide");
const overlaySelect = createFileSelect("Load overlay");

fileSelectors.append(
  slideSelect,
  overlaySelect,
);

const fileActions = document.createElement("div");
fileActions.className = "viewer-file-actions";

function createFileActionButton(label) {
  const button = document.createElement("button");
  button.type = "button";
  button.textContent = label;
  return button;
}

const clearSlideButton =
  createFileActionButton("Clear Slide");

const clearOverlaysButton =
  createFileActionButton("Clear Overlays");

clearSlideButton.disabled = true;
clearOverlaysButton.disabled = true;

fileActions.append(
  clearSlideButton,
  clearOverlaysButton,
);

viewerFiles.append(
  fileSelectors,
  fileActions,
);

function populateFileSelect(
  select,
  files,
  placeholder,
) {
  select.setFiles(files, placeholder);
}

function getMatchingOverlays(slidePath) {
  const slideStem = getFileStem(slidePath);

  return configuredOverlays.files.filter((file) => {
    const fileName =
      file.name.split(/[\\/]/).pop() ?? file.name;

    return fileName.includes(slideStem);
  });
}

function updateOverlaySelect() {
  if (configuredOverlays.directory === null) {
    populateFileSelect(
      overlaySelect,
      [],
      "No overlay directory configured",
    );
    return;
  }

  if (currentSlidePath === null) {
    populateFileSelect(
      overlaySelect,
      [],
      "Select slide first",
    );
    return;
  }

  const matchingOverlays =
    getMatchingOverlays(currentSlidePath);

  populateFileSelect(
    overlaySelect,
    matchingOverlays,
    matchingOverlays.length === 0
      ? "No matching overlays"
      : "Load overlay",
  );
}

const configuredSlides =
  await getConfiguredFiles("slide");

const configuredOverlays =
  await getConfiguredFiles("overlay");

populateFileSelect(
  slideSelect,
  configuredSlides.files,
  configuredSlides.directory === null
    ? "No slide directory configured"
    : "Select slide",
);

updateOverlaySelect();

function updateSlideSelect(filePath) {
  slideSelect.value = filePath;
}

function updateFileActionState() {
  const hasSlide = currentSlideInfo !== null;
  const hasOverlays =
    Object.keys(overlayLayers).length > 0;
  const hasMatchingOverlays =
    currentSlidePath !== null &&
    getMatchingOverlays(currentSlidePath).length > 0;

  clearSlideButton.disabled = !hasSlide;
  clearOverlaysButton.disabled =
    !hasSlide || !hasOverlays;

  slideSelect.disabled =
    configuredSlides.files.length === 0;

  overlaySelect.disabled =
    !hasSlide || !hasMatchingOverlays;
}

function setFileActionsBusy(busy) {
  if (!busy) {
    updateFileActionState();
    return;
  }

  slideSelect.disabled = true;
  overlaySelect.disabled = true;
  clearSlideButton.disabled = true;
  clearOverlaysButton.disabled = true;
}

slideSelect.addEventListener("change", async (event) => {
  const filePath =
    event.detail ?? slideSelect.value;

  if (filePath === "") {
    return;
  }

  setFileActionsBusy(true);

  try {
    await switchSlide(filePath);
    overlaySelect.value = "";
  } catch (error) {
    console.error(error);
  } finally {
    setFileActionsBusy(false);
  }
});

overlaySelect.addEventListener(
  "change",
  async (event) => {
    const filePath =
      event.detail ?? overlaySelect.value;

    if (filePath === "") {
      return;
    }

    setFileActionsBusy(true);

    try {
      await loadOverlay(filePath);
      overlaySelect.value = "";
    } catch (error) {
      console.error(error);
    } finally {
      setFileActionsBusy(false);
    }
  },
);

clearSlideButton.addEventListener("click", async () => {
  setFileActionsBusy(true);

  try {
    await removeSlide();

    slideSelect.value = "";
    overlaySelect.value = "";
    updateOverlaySelect();
  } catch (error) {
    console.error(error);
  } finally {
    setFileActionsBusy(false);
  }
});

clearOverlaysButton.addEventListener(
  "click",
  async () => {
    setFileActionsBusy(true);

    try {
      await clearOverlays();
      overlaySelect.value = "";
    } catch (error) {
      console.error(error);
    } finally {
      setFileActionsBusy(false);
    }
  },
);

updateFileActionState();

function getFileStem(filePath) {
  const fileName = filePath.split(/[\\/]/).pop() ?? filePath;
  const extensionIndex = fileName.lastIndexOf(".");

  if (extensionIndex <= 0) {
    return fileName;
  }

  return fileName.slice(0, extensionIndex);
}

// Dynamic slide loading
const params = new URLSearchParams(window.location.search);

const slidePath =
  params.get("slide") ??
  (layersData.length === 0
    ? configuredSlides.files[0]?.path ?? null
    : null);

if (slidePath !== null) {
  currentSlidePath = slidePath;
  sessionId = await createSession();
  const slideInfo = await loadSlide(slidePath);

  currentSlideInfo = slideInfo;
  updateSlideSelect(slidePath);
  updateOverlaySelect();

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

updateFileActionState();

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
    code: "ZoomifyProjectionEmpty",
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

  // OpenLayers 10.10 internally uses 300 scroll-delta
  // units per zoom level, but does not expose a public
  // sensitivity option.
  interaction.deltaPerZoom_ =
    preset.deltaPerZoom;

  interaction.setActive(
    slideLayer.getSource() !== null,
  );

  return interaction;
}

let mouseWheelZoomInteraction =
  createMouseWheelZoomInteraction();

function createZoomControl() {
  return new Zoom({
    delta: Number(zoomButtonStepSelect.value),
  });
}

let zoomControlInstance =
  createZoomControl();

const map = new Map({
  target: mapElement,
  layers,
  view,
  controls: defaultControls({
    zoom: false,
    rotate: false,
  }).extend([
    zoomControlInstance,
  ]),
  interactions: defaultInteractions({
    mouseWheelZoom: false,
  }).extend([
    mouseWheelZoomInteraction,
  ]),
});

function updateMouseWheelZoomSensitivity() {
  map.removeInteraction(
    mouseWheelZoomInteraction,
  );

  mouseWheelZoomInteraction =
    createMouseWheelZoomInteraction();

  map.addInteraction(
    mouseWheelZoomInteraction,
  );
}

mouseWheelZoomSensitivitySelect.addEventListener(
  "change",
  () => {
    updateMouseWheelZoomSensitivity();
  },
);

// Zoom level
let zoomControl = zoomControlInstance.element;

let zoomOutButton =
  zoomControl.querySelector(".ol-zoom-out");

if (zoomOutButton === null) {
  throw new Error(
    "The OpenLayers zoom control could not be found.",
  );
}

const zoomLevel = document.createElement("input");

zoomLevel.type = "number";
zoomLevel.className = "ol-zoom-level";
zoomLevel.step = "1";
zoomLevel.setAttribute("aria-label", "Zoom level");
zoomLevel.title = "Zoom level";

zoomControl.insertBefore(zoomLevel, zoomOutButton);

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
  const zoom = Number.parseFloat(zoomLevel.value);

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

zoomLevel.addEventListener("keydown", (event) => {
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
});

updateZoomLevel();

function resetView() {
  const source = slideLayer.getSource();

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

resetViewButton.addEventListener("click", () => {
  resetView();
});

// Scalebar
const scaleBarWidths = {
  small: 70,
  default: 100,
  large: 140,
};

function createScaleLineControl() {
  const minWidth =
    scaleBarWidths[scaleBarSizeSelect.value] ??
    scaleBarWidths.default;

  return new ScaleLine({
    units: scaleBarUnitsSelect.value,
    minWidth,
  });
}

let scaleLineControl = createScaleLineControl();

map.addControl(scaleLineControl);

function updateScaleBarVisibility() {
  const hasSlide = slideLayer.getSource() !== null;

  scaleLineControl.element.classList.toggle(
    "viewer-control-hidden",
    !hasSlide || !scaleBarEnabled,
  );
}

scaleBarEnabledInput.addEventListener("change", () => {
  scaleBarEnabled = scaleBarEnabledInput.checked;
  updateScaleBarVisibility();
});

function updateScaleBarColour() {
  const colour = scaleBarColourInput.value;

  const scaleLineInner = scaleLineControl.element.querySelector(
    ".ol-scale-line-inner",
  );

  if (scaleLineInner === null) {
    return;
  }

  scaleLineInner.style.color = colour;
  scaleLineInner.style.borderColor = colour;
}

scaleBarColourInput.addEventListener("input", () => {
  updateScaleBarColour();
  updateScaleBarOpacity();
});

function updateScaleBarOpacity() {
  const opacity =
    Number(scaleBarOpacityInput.value) / 100;

  const scaleBarColour =
    hexToRgb(scaleBarColourInput.value);

  if (scaleBarColour === null) {
    return;
  }

  const contrastColour =
    getContrastingColour(scaleBarColour);

  const backgroundColour =
    contrastColour === "#ffffff"
      ? { r: 255, g: 255, b: 255 }
      : { r: 17, g: 17, b: 17 };

  scaleLineControl.element.style.backgroundColor =
    toRgba(backgroundColour, opacity);

  scaleBarOpacityValue.textContent =
    `${scaleBarOpacityInput.value}%`;
}

scaleBarOpacityInput.addEventListener("input", () => {
  updateScaleBarOpacity();
});

function updateScaleBarSize() {
  map.removeControl(scaleLineControl);

  scaleLineControl = createScaleLineControl();

  map.addControl(scaleLineControl);

  window.scaleLineControl = scaleLineControl;

  updateScaleBarVisibility();
  updateScaleBarColour();
  updateScaleBarOpacity();
}

scaleBarSizeSelect.addEventListener("change", () => {
  updateScaleBarSize();
});

scaleBarUnitsSelect.addEventListener("change", () => {
  scaleLineControl.setUnits(scaleBarUnitsSelect.value);
});

updateScaleBarColour();
updateScaleBarOpacity();

const overviewLayer = new TileLayer();

if (baseSource !== null) {
  overviewLayer.setSource(baseSource);
}

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

function getOverviewMapSize() {
  return (
    overviewMapSizes[overviewMapSizeSelect.value] ??
    overviewMapSizes.default
  );
}

function createOverviewView(overviewProjection, overviewExtent) {
  const overviewMapSize = getOverviewMapSize();

  const center = [
    (overviewExtent[0] + overviewExtent[2]) / 2,
    (overviewExtent[1] + overviewExtent[3]) / 2,
  ];

  const width = overviewExtent[2] - overviewExtent[0];
  const height = overviewExtent[3] - overviewExtent[1];

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

function updateOverviewMapSize() {
  const size = getOverviewMapSize();

  overviewMapControl.element.style.setProperty(
    "--overview-map-width",
    `${size.width}px`,
  );

  overviewMapControl.element.style.setProperty(
    "--overview-map-height",
    `${size.height}px`,
  );

  overviewMap.updateSize();

  const source = overviewLayer.getSource();

  if (source !== null) {
    const currentProjection =
      map.getView().getProjection();

    const currentExtent =
      source.getTileGrid().getExtent();

    overviewMap.setView(
      createOverviewView(
        currentProjection,
        currentExtent,
      ),
    );
  }

  overviewMap.renderSync();
}

overviewMapSizeSelect.addEventListener(
  "change",
  () => {
    updateOverviewMapSize();
  },
);

updateOverviewMapSize();

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

const bottomControlsGroup = document.createElement("div");

bottomControlsGroup.className =
  "bottom-controls-group ol-unselectable";

viewerApp.append(bottomControlsGroup);

bottomControlsGroup.append(
  resetViewControl,
  zoomControl,
  fullscreen.element,
);

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

  setViewerEnabled(
    slideLayer.getSource() !== null,
  );

  updateControlVisibility();
}

zoomButtonStepSelect.addEventListener(
  "change",
  () => {
    updateZoomButtonStep();
  },
);

// Layer switcher
const layerSwitcher = new LayerSwitcher();

map.addControl(layerSwitcher);

// Graticule
const gridSpacingValues = {
  fine: 32,
  default: 64,
  coarse: 128,
};

function getGridSpacing() {
  return (
    gridSpacingValues[gridSpacingSelect.value] ??
    gridSpacingValues.default
  );
}

const graticuleMargin = 64;

const gridThemeColours = {
  light: {
    line: { r: 255, g: 255, b: 255 },
    label: "rgba(255, 255, 255, 1)",
    outline: "rgba(20, 20, 20, 1)",
  },

  dark: {
    line: { r: 20, g: 20, b: 20 },
    label: "rgba(20, 20, 20, 1)",
    outline: "rgba(255, 255, 255, 1)",
  },

  "light-contrast": {
    line: { r: 0, g: 170, b: 200 },
    label: "rgba(0, 170, 200, 1)",
    outline: "rgba(20, 20, 20, 1)",
  },

  "dark-contrast": {
    line: { r: 145, g: 55, b: 0 },
    label: "rgba(145, 55, 0, 1)",
    outline: "rgba(255, 255, 255, 1)",
  },
};

function getGridTheme() {
  if (gridThemeSelect.value !== "default") {
    return gridThemeSelect.value;
  }

  if (themeSelect.value === "light") {
    return "light";
  }

  if (themeSelect.value === "high-contrast") {
    return "dark-contrast";
  }

  return "dark";
}

const graticuleTextStyle = new Text({
  font: "12px Calibri,sans-serif",
  fill: new Fill({
    color: "rgba(0, 0, 0, 1)",
  }),
  stroke: new Stroke({
    color: "rgba(255, 255, 255, 1)",
    width: 3,
  }),
});

const graticuleStyle = new Style({
  stroke: new Stroke({
    color: "rgba(0, 0, 0, 0.5)",
    width: 1,
  }),
  text: graticuleTextStyle,
});

function updateGridAppearance() {
  const gridTheme = getGridTheme();

  const colours =
    gridThemeColours[gridTheme] ??
    gridThemeColours.dark;

  const opacity =
    Number(gridOpacityInput.value) / 100;

  const gridStroke = graticuleStyle.getStroke();
  const gridText = graticuleTextStyle;

  gridStroke.setColor(
    toRgba(colours.line, opacity),
  );

  gridText
    .getFill()
    .setColor(colours.label);

  gridText
    .getStroke()
    .setColor(colours.outline);

  gridOpacityValue.textContent =
    `${gridOpacityInput.value}%`;

  map.renderSync();
}

function updateGridLabels() {
  graticuleStyle.setText(
    gridLabelsVisibleInput.checked
      ? graticuleTextStyle
      : null,
  );

  map.renderSync();
}

gridThemeSelect.addEventListener("change", () => {
  updateGridAppearance();
});

gridOpacityInput.addEventListener("input", () => {
  updateGridAppearance();
});

gridLabelsVisibleInput.addEventListener("change", () => {
  updateGridLabels();
});

// Create graticules for the active slide projection.
function createGraticule(graticuleProjection) {
  return new Graticule({
    projection: graticuleProjection,
    margin: graticuleMargin,
    style: graticuleStyle,
    spacing: getGridSpacing(),
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
const screenSpaceGraticuleMargin = graticuleMargin;

function createScreenSpaceGraticule(graticuleProjection) {
  const spacing = getGridSpacing();

  return new Graticule({
    projection: graticuleProjection.getCode(),
    spacing,
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
          spacing,
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

updateGridAppearance();
updateGridLabels();

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

const viewerToolsGroup = document.createElement("div");

viewerToolsGroup.className =
  "viewer-tools-group ol-unselectable";

map
  .getOverlayContainerStopEvent()
  .append(viewerToolsGroup);

viewerToolsGroup.append(
  rotate.element,
  graticuleToggle.element,
  screenSpaceGraticuleToggle.element,
);

function updateGridSpacing() {
  const graticuleWasActive =
    graticuleToggle.getActive();

  const screenSpaceGraticuleWasActive =
    screenSpaceGraticuleToggle.getActive();

  graticule.setMap(null);
  screenSpaceGraticule.setMap(null);

  const projection = map.getView().getProjection();

  graticule = createGraticule(projection);

  screenSpaceGraticule =
    createScreenSpaceGraticule(projection);

  if (graticuleWasActive) {
    graticule.setMap(map);
  }

  if (screenSpaceGraticuleWasActive) {
    screenSpaceGraticule.setMap(map);
  }

  window.graticule = graticule;
  window.screenSpaceGraticule =
    screenSpaceGraticule;

  map.renderSync();
}

gridSpacingSelect.addEventListener("change", () => {
  updateGridSpacing();
});

function updateControlVisibility() {
  const hasSlide = slideLayer.getSource() !== null;

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

  graticuleToggle.element.classList.toggle(
    "viewer-control-hidden",
    !graticuleVisibleInput.checked,
  );

  screenSpaceGraticuleToggle.element.classList.toggle(
    "viewer-control-hidden",
    !screenSpaceGraticuleVisibleInput.checked,
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
    !hasSlide || !mousePositionVisibleInput.checked,
  );

  overviewMapControl.element.classList.toggle(
    "viewer-control-hidden",
    !hasSlide || !overviewMapVisibleInput.checked,
  );

  if (!graticuleVisibleInput.checked) {
    graticuleToggle.setActive(false);
    graticuleToggle.element.classList.remove("active");
    graticule.setMap(null);
  }

  if (!screenSpaceGraticuleVisibleInput.checked) {
    screenSpaceGraticuleToggle.setActive(false);
    screenSpaceGraticuleToggle.element.classList.remove(
      "active",
    );
    screenSpaceGraticule.setMap(null);
  }

  if (hasSlide && overviewMapVisibleInput.checked) {
    requestAnimationFrame(() => {
      overviewMap.updateSize();
      overviewMap.renderSync();
    });
  }
}

for (const input of [
  zoomVisibleInput,
  zoomLevelVisibleInput,
  rotationVisibleInput,
  graticuleVisibleInput,
  screenSpaceGraticuleVisibleInput,
  resetViewVisibleInput,
  fullscreenVisibleInput,
  mousePositionVisibleInput,
  overviewMapVisibleInput,
]) {
  input.addEventListener("change", () => {
    updateControlVisibility();
  });
}

function resetSettingsToDefaults() {
  themeSelect.value = "dark";
  overviewMapSizeSelect.value = "default";
  mouseWheelZoomSensitivitySelect.value = "default";
  zoomButtonStepSelect.value = "1";
  gridThemeSelect.value = "default";
  gridOpacityInput.value = "50";
  gridSpacingSelect.value = "default";
  gridLabelsVisibleInput.checked = true;
  controlOpacityInput.value = "100";

  for (const input of [
    zoomVisibleInput,
    zoomLevelVisibleInput,
    rotationVisibleInput,
    graticuleVisibleInput,
    screenSpaceGraticuleVisibleInput,
    resetViewVisibleInput,
    fullscreenVisibleInput,
    mousePositionVisibleInput,
    overviewMapVisibleInput,
  ]) {
    input.checked = true;
  }

  scaleBarEnabledInput.checked = true;
  scaleBarEnabled = true;

  scaleBarColourInput.value = "#ffffff";
  scaleBarOpacityInput.value = "100";
  scaleBarSizeSelect.value = "default";
  scaleBarUnitsSelect.value = "metric";

  updateControlAppearance();
  updateGridAppearance();
  updateGridSpacing();
  updateGridLabels();
  updateControlVisibility();
  updateOverviewMapSize();
  updateMouseWheelZoomSensitivity();
  updateZoomButtonStep();

  updateScaleBarSize();
  scaleLineControl.setUnits(scaleBarUnitsSelect.value);

  updateScaleBarVisibility();
  updateScaleBarColour();
  updateScaleBarOpacity();

  try {
    window.localStorage.removeItem(
      settingsStorageKey,
    );
  } catch {
    // The defaults still apply if storage is unavailable.
  }
}

resetDefaultsButton.addEventListener("click", () => {
  resetSettingsToDefaults();
});

for (const input of [
  themeSelect,
  gridThemeSelect,
  gridSpacingSelect,
  gridLabelsVisibleInput,
  zoomVisibleInput,
  zoomLevelVisibleInput,
  rotationVisibleInput,
  graticuleVisibleInput,
  screenSpaceGraticuleVisibleInput,
  resetViewVisibleInput,
  fullscreenVisibleInput,
  mousePositionVisibleInput,
  overviewMapVisibleInput,
  overviewMapSizeSelect,
  mouseWheelZoomSensitivitySelect,
  zoomButtonStepSelect,

  scaleBarEnabledInput,
  scaleBarSizeSelect,
  scaleBarUnitsSelect,
]) {
  input.addEventListener("change", () => {
    saveSettings();
  });
}

for (const input of [
  controlOpacityInput,
  gridOpacityInput,
  scaleBarColourInput,
  scaleBarOpacityInput,
]) {
  input.addEventListener("input", () => {
    saveSettings();
  });
}

// Enable or hide controls that require a loaded slide.
function setViewerEnabled(enabled) {
  const zoomInButton = zoomControl.querySelector(".ol-zoom-in");
  const zoomOutButton = zoomControl.querySelector(".ol-zoom-out");
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

  scaleLineControl.element.classList.toggle(
    "viewer-control-hidden",
    !enabled || !scaleBarEnabled,
  );
  mousePositionControl.element.classList.toggle(
    "viewer-control-hidden",
    !enabled || !mousePositionVisibleInput.checked,
  );
  overviewMapControl.element.classList.toggle(
    "viewer-control-hidden",
    !enabled || !overviewMapVisibleInput.checked,
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
updateControlVisibility();

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

  annotationLayerNames.clear();

  updateLayerEditor();
  updateFileActionState();
}

async function clearOverlays() {
  const response = await fetch("/tileserver/clear_overlays", {
    method: "PUT",
  });

  if (!response.ok) {
    throw new Error("Failed to clear overlays.");
  }

  clearOverlayLayers();
  updateFileActionState();
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
  layersData.length = 0;

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
  updateFileActionState();
}

// Slide switching
async function switchSlide(slidePath) {
  if (sessionId === null) {
    throw new Error("Dynamic slide switching requires a TileServer session.");
  }

  clearOverlayLayers();

  const slideInfo = await loadSlide(slidePath);
  currentSlideInfo = slideInfo;
  currentSlidePath = slidePath;
  updateSlideSelect(slidePath);
  updateOverlaySelect();
  updateFileActionState();

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
      id: "slide",
      name: getFileStem(currentSlidePath ?? "slide"),
      layer: slideLayer,
    });
  }

  const overlayEntries = Object.entries(overlayLayers)
    .map(([layerName, layer]) => ({
      id: layerName,
      name: layerName,
      layer,
    }))
    .sort(
      (a, b) =>
        (a.layer.getZIndex() ?? 0) -
        (b.layer.getZIndex() ?? 0),
    );

  entries.push(...overlayEntries);

  return entries;
}

function moveLayer(layerId, direction) {
  if (layerId === "slide") {
    return;
  }

  const entries = getLayerEditorEntries().filter(
    (entry) => entry.id !== "slide",
  );

  const index = entries.findIndex(
    (entry) => entry.id === layerId,
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

  const overlayEntries = entries.filter(
    (entry) => entry.id !== "slide",
  );

  if (entries.length === 0) {
    const empty = document.createElement("div");
    empty.className = "layer-editor-empty";
    empty.textContent = "No layers loaded";

    layerEditorList.appendChild(empty);

    return;
  }

  entries.forEach(({ id: layerId, name: layerName, layer }) => {
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
    name.title = layerName;

    header.append(
      visibility,
      name,
    );

    if (layerId !== "slide") {
      const overlayIndex = overlayEntries.findIndex(
        (entry) => entry.id === layerId,
      );

      const order = document.createElement("div");
      order.className = "layer-editor-order";

      const moveUp = document.createElement("button");
      moveUp.type = "button";
      moveUp.title = "Move layer up";
      moveUp.innerHTML =
        '<i class="fas fa-chevron-up"></i>';
      moveUp.disabled = overlayIndex === 0;

      moveUp.addEventListener("click", () => {
        moveLayer(layerId, "up");
      });

      const moveDown = document.createElement("button");
      moveDown.type = "button";
      moveDown.title = "Move layer down";
      moveDown.innerHTML =
        '<i class="fas fa-chevron-down"></i>';
      moveDown.disabled =
        overlayIndex === overlayEntries.length - 1;

      moveDown.addEventListener("click", () => {
        moveLayer(layerId, "down");
      });

      const remove = document.createElement("button");
      remove.type = "button";
      remove.title = `Remove ${layerName}`;
      remove.innerHTML =
        '<i class="fas fa-times"></i>';

      remove.addEventListener("click", () => {
        removeOverlay(layerId).catch((error) => {
          console.error(error);
        });
      });

      order.append(
        moveUp,
        moveDown,
        remove,
      );

      header.appendChild(order);
    }

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

  const isAnnotation = ["db", "dat", "geojson"].includes(
    extension,
  );

  const layerName = getFileStem(overlayPath);

  if (layerName === "slide") {
    throw new Error(
      'The overlay name "slide" is reserved.',
    );
  }

  const formData = new FormData();
  formData.append("overlay_path", overlayPath);
  formData.append("layer_name", layerName);

  const response = await fetch("/tileserver/overlay", {
    method: "PUT",
    body: formData,
  });

  if (!response.ok) {
    throw new Error(`Failed to load overlay: ${overlayPath}`);
  }

  const result = await response.json();

  if (isAnnotation) {
    annotationLayerNames.add(layerName);
  } else {
    annotationLayerNames.delete(layerName);
  }

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
    // Replace an existing layer with the same filename stem.
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
  updateFileActionState();

  return result;
}

async function removeOverlay(layerName) {
  const overlayLayer = overlayLayers[layerName];

  if (overlayLayer === undefined) {
    throw new Error(`Overlay is not loaded: ${layerName}`);
  }

  const source = overlayLayer.getSource();

  overlayLayer.setVisible(false);
  overlayLayer.setSource(null);
  map.removeLayer(overlayLayer);

  const response = await fetch(
    `/tileserver/overlay/${encodeURIComponent(layerName)}`,
    {
      method: "DELETE",
    },
  );

  if (!response.ok) {
    overlayLayer.setSource(source);
    overlayLayer.setVisible(true);
    map.addLayer(overlayLayer);

    throw new Error(`Failed to remove overlay: ${layerName}`);
  }

  const layerIndex = layers.indexOf(overlayLayer);

  if (layerIndex !== -1) {
    layers.splice(layerIndex, 1);
  }

  annotationLayerNames.delete(layerName);
  delete overlayLayers[layerName];

  updateLayerEditor();
  updateFileActionState();
}

async function setAnnotationColors(colorMap) {
  if (annotationLayerNames.size === 0) {
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

  for (const layerName of annotationLayerNames) {
    const overlayLayer = overlayLayers[layerName];

    if (overlayLayer === undefined) {
      continue;
    }

    const source = new Zoomify({
      url:
        `/tileserver/layer/${encodeURIComponent(layerName)}/` +
        `${sessionId}/zoomify/` +
        `{TileGroup}/{z}-{x}-{y}@1x.jpg?v=${overlayVersion}`,
      size: currentSlideInfo.slide_dimensions,
      crossOrigin: "anonymous",
      zDirection: -1,
    });

    overlayLayer.setSource(source);
  }
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
