import "ol/ol.css";
import "ol-ext/dist/ol-ext.css";
import "./style.css";

import { defaults as defaultControls } from "ol/control/defaults.js";
import TileLayer from "ol/layer/Tile.js";
import Map from "ol/Map.js";
import Projection from "ol/proj/Projection.js";
import { addProjection } from "ol/proj.js";
import Zoomify from "ol/source/Zoomify.js";
import View from "ol/View.js";

import LayerSwitcher from "ol-ext/control/LayerSwitcher.js";

import { defaults as defaultInteractions } from "ol/interaction/defaults.js";
import {
    clearOverlays as clearTileServerOverlays,
    createSession,
    getConfiguredFiles,
    loadOverlay as loadTileServerOverlay,
    loadSlide,
    removeOverlay as removeTileServerOverlay,
    removeSlide as removeTileServerSlide,
    setAnnotationColors as setTileServerAnnotationColors,
} from "./api/tileserver.js";
import {
    createMapControlsController,
} from "./controls/map-controls.js";
import {
    createGridController,
} from "./controls/grid.js";
import {
    createScaleBarController,
} from "./controls/scale-bar.js";
import {
    createOverviewMapController,
} from "./controls/overview-map.js";
import {
    createFilesPanelController,
} from "./panels/files.js";
import {
    createLayersPanelController,
} from "./panels/layers.js";
import {
    createSettingsPanelController,
} from "./panels/settings.js";
import { getFileStem } from "./utils/paths.js";

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

const scaleBarThemeColours = {
    dark: "#ffffff",
    light: "#000000",
    "high-contrast": "#ffffff",
};

// Load persisted settings before creating controls that depend on them.
// Settings events are bound after all feature controllers are created.
const settingsPanelController =
    createSettingsPanelController({
        viewerApp,
        panel: settingsPanel,
        toggle: settingsToggle,
        closeButton: settingsCloseButton,
        tabs: settingsTabs,
        tabPanels:
            settingsTabPanels,
        resetDefaultsButton,
        themeSelect,
        controlOpacityInput,
        controlOpacityValue,
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
        gridThemeSelect,
        gridOpacityInput,
        gridSpacingSelect,
        gridLabelsVisibleInput,
        scaleBarEnabledInput,
        scaleBarColourInput,
        scaleBarOpacityInput,
        scaleBarSizeSelect,
        scaleBarUnitsSelect,

        onThemeChange() {
            scaleBarColourInput.value =
                scaleBarThemeColours[
                    themeSelect.value
                ] ??
                scaleBarThemeColours.dark;

            scaleBarController.updateColour();
            scaleBarController.updateOpacity();

            gridController.updateAppearance();
        },

        onControlVisibilityChange() {
            updateControlVisibility();
        },

        onReset() {
            resetSettingsToDefaults();
        },
    });

settingsPanelController.load();

settingsPanelController.updateAppearance();

const layersPanelController =
    createLayersPanelController({
        panel: layerEditor,
        toggle: layerEditorToggle,
        list: layerEditorList,

        getSlideLayer: () => slideLayer,

        getCurrentSlidePath: () =>
            currentSlidePath,

        getOverlayLayers: () =>
            overlayLayers,

        onRemoveLayer: (layerId) =>
            removeOverlay(layerId),

        onOpen() {
            filesPanelController.setOpen(false);
        },
    });

let layersData = JSON.parse(mapElement.dataset.layers ?? "[]");
let sessionId = null;
let slideVersion = Date.now();
let overlayVersion = Date.now();
let currentSlideInfo = null;
let currentSlidePath = null;
const overlayLayers = {};
const annotationLayerNames = new Set();

const configuredSlides =
    await getConfiguredFiles("slide");

const configuredOverlays =
    await getConfiguredFiles("overlay");

const filesPanelController =
    createFilesPanelController({
        panel: viewerPanel,
        toggle: viewerPanelToggle,
        container: viewerFiles,
        configuredSlides,
        configuredOverlays,

        getCurrentSlidePath: () =>
            currentSlidePath,

        hasSlide: () =>
            currentSlideInfo !== null,

        hasOverlays: () =>
            Object.keys(
                overlayLayers,
            ).length > 0,

        onSlideSelected: (filePath) =>
            switchSlide(filePath),

        onOverlaySelected: (filePath) =>
            loadOverlay(filePath),

        onClearSlide: () =>
            removeSlide(),

        onClearOverlays: () =>
            clearOverlays(),

        onOpen() {
            layersPanelController.setOpen(
                false,
            );
        },
    });

filesPanelController.setOpen(true);

// Resolve and load the initial slide for dynamic TileServer mode.
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
    filesPanelController.setSlide(
        slidePath,
    );
    filesPanelController.updateOverlaySelect();

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

filesPanelController.updateActionState();

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

// Pad the view extent so users can pan slightly beyond the slide.
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
    controls: defaultControls({
        zoom: false,
        rotate: false,
    }),
    interactions: defaultInteractions({
        mouseWheelZoom: false,
    }),
});

const mapControlsController =
    createMapControlsController({
        map,
        viewerApp,
        getSlideSource: () =>
            slideLayer.getSource(),
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
    });

const {
    fullscreen,
    mousePositionControl,
    rotate,
} = mapControlsController;

// Scale bar
let scaleLineControl = null;

const scaleBarController =
    createScaleBarController({
        map,
        hasSlide: () => slideLayer.getSource() !== null,
        enabledInput: scaleBarEnabledInput,
        colourInput: scaleBarColourInput,
        opacityInput: scaleBarOpacityInput,
        opacityValue: scaleBarOpacityValue,
        sizeSelect: scaleBarSizeSelect,
        unitsSelect: scaleBarUnitsSelect,
        onControlChange(control) {
            scaleLineControl = control;
            window.scaleLineControl = control;
        },
    });

// Overview map
const overviewMapController =
    createOverviewMapController({
        map,
        source: baseSource,
        projection,
        extent,
        sizeSelect: overviewMapSizeSelect,
        visibleInput: overviewMapVisibleInput,
        hasSlide: () => slideLayer.getSource() !== null,
    });

const overviewMapControl =
    overviewMapController.control;

// Layer switcher
const layerSwitcher = new LayerSwitcher();

map.addControl(layerSwitcher);

// Grid controls
let graticule = null;
let screenSpaceGraticule = null;

const gridController =
    createGridController({
        map,
        projection,
        themeSelect,
        gridThemeSelect,
        gridOpacityInput,
        gridOpacityValue,
        gridSpacingSelect,
        gridLabelsVisibleInput,
        graticuleVisibleInput,
        screenSpaceGraticuleVisibleInput,

        onGraticulesChange(
            nextGraticule,
            nextScreenSpaceGraticule,
        ) {
            graticule = nextGraticule;
            screenSpaceGraticule =
                nextScreenSpaceGraticule;

            window.graticule = graticule;
            window.screenSpaceGraticule =
                screenSpaceGraticule;
        },
    });

const {
    graticuleToggle,
    screenSpaceGraticuleToggle,
} = gridController;

const viewerToolsGroup =
    document.createElement("div");

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

function updateControlVisibility() {
    mapControlsController.updateVisibility();
    gridController.updateVisibility();
    overviewMapController.updateVisibility();
}

function resetSettingsToDefaults() {
    settingsPanelController.resetValues();

    settingsPanelController.updateAppearance();

    gridController.updateAppearance();
    gridController.updateSpacing();
    gridController.updateLabels();

    updateControlVisibility();

    overviewMapController.updateSize();

    mapControlsController.updateMouseWheelZoomSensitivity();

    mapControlsController.updateZoomButtonStep();

    scaleBarController.updateSize();
    scaleBarController.updateUnits();

    scaleBarController.updateVisibility();
    scaleBarController.updateColour();
    scaleBarController.updateOpacity();

    settingsPanelController.clearSavedSettings();
}

settingsPanelController.bindEvents();

// Enable or hide controls that require a loaded slide.
function setViewerEnabled(enabled) {
    mapControlsController.setViewerEnabled(enabled);
    gridController.setViewerEnabled(enabled);
    scaleBarController.setViewerEnabled(enabled);
    overviewMapController.setViewerEnabled(enabled);
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
    mapControlsController.updateZoomLevel();
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

    layersPanelController.render();
    filesPanelController.updateActionState();
}

async function clearOverlays() {
    await clearTileServerOverlays();

    clearOverlayLayers();
    filesPanelController.updateActionState();
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

    await removeTileServerSlide();

    clearOverlayLayers();

    currentSlidePath = null;
    currentSlideInfo = null;
    layersData.length = 0;

    slideVersion += 1;
    overlayVersion += 1;

    slideLayer.setSource(null);
    overviewMapController.setSource(null);

    layersPanelController.render();

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

    overviewMapController.setView(
        emptyProjection,
        emptyExtent,
    );

    gridController.setProjection(
        emptyProjection,
        {
            preserveActive: false,
        },
    );

    const url = new URL(window.location.href);
    url.search = "";
    url.hash = "";

    window.history.replaceState({}, "", url);

    setViewerEnabled(false);
    mapControlsController.updateZoomLevel();
    filesPanelController.updateActionState();
}

// Replace the current slide and rebuild its source, projection and view.
async function switchSlide(slidePath) {
    if (sessionId === null) {
        throw new Error("Dynamic slide switching requires a TileServer session.");
    }

    clearOverlayLayers();

    const slideInfo = await loadSlide(slidePath);
    currentSlideInfo = slideInfo;
    currentSlidePath = slidePath;

    filesPanelController.setSlide(
        slidePath,
    );

    filesPanelController.updateOverlaySelect();

    filesPanelController.updateActionState();

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

    // Rebuild the view for the new slide extent and projection.
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

    overviewMapController.setView(
        newProjection,
        newExtent,
    );

    gridController.setProjection(
        newProjection,
    );

    slideLayer.setSource(source);
    overviewMapController.setSource(source);

    layersPanelController.render();

    setViewerEnabled(true);
    updateUrlState();
    mapControlsController.updateZoomLevel();
}

layersPanelController.render();

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

    const result = await loadTileServerOverlay(
        overlayPath,
        layerName,
    );

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

    layersPanelController.render();

    filesPanelController.updateActionState();

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

    try {
        await removeTileServerOverlay(layerName);
    } catch (error) {
        // Restore the frontend layer if the TileServer removal fails.
        overlayLayer.setSource(source);
        overlayLayer.setVisible(true);
        map.addLayer(overlayLayer);

        throw error;
    }

    const layerIndex = layers.indexOf(overlayLayer);

    if (layerIndex !== -1) {
        layers.splice(layerIndex, 1);
    }

    annotationLayerNames.delete(layerName);
    delete overlayLayers[layerName];

    layersPanelController.render();
    filesPanelController.updateActionState();
}

async function setAnnotationColors(colorMap) {
    if (annotationLayerNames.size === 0) {
        throw new Error("No annotation overlay is loaded.");
    }

    await setTileServerAnnotationColors(colorMap);

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
