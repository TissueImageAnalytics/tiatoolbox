import "ol/ol.css";
import "ol-ext/dist/ol-ext.css";
import "./style.css";

import { defaults as defaultControls } from "ol/control/defaults.js";
import TileLayer from "ol/layer/Tile.js";
import OlMap from "ol/Map.js";
import Projection from "ol/proj/Projection.js";
import { addProjection } from "ol/proj.js";
import Zoomify from "ol/source/Zoomify.js";
import View from "ol/View.js";

import LayerSwitcher from "ol-ext/control/LayerSwitcher.js";

import { defaults as defaultInteractions } from "ol/interaction/defaults.js";
import {
    clearAnnotationSecondaryMapper,
    clearOverlays as clearTileServerOverlays,
    createSession,
    getConfiguredFiles,
    loadOverlay as loadTileServerOverlay,
    loadSlide,
    removeOverlay as removeTileServerOverlay,
    removeSlide as removeTileServerSlide,
    getAnnotationColors as getTileServerAnnotationColors,
    getAnnotationAtPoint,
    getAnnotationProperties,
    getAnnotationPropertyValues,
    setAnnotationFilter as setTileServerAnnotationFilter,
    setAnnotationColors as setTileServerAnnotationColors,
    setAnnotationOpacities as setTileServerAnnotationOpacities,
    setAnnotationMapper,
    setAnnotationProperty,
    setAnnotationPropertyRange,
    setAnnotationSecondaryMapper,
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
    createAnnotationsPanelController,
} from "./panels/annotations.js";
import {
    createSettingsPanelController,
} from "./panels/settings.js";
import {
    assignAnnotationColours,
    createAnnotationColourConfig,
    mergeAnnotationColourConfig,
    parseAnnotationColourConfig,
} from "./utils/annotation-colours.js";
import {
    getAnnotationFilter,
} from "./utils/annotation-filters.js";
import { hexToRgb } from "./utils/colours.js";
import {
    getFiniteNumberRange,
} from "./utils/numbers.js";
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

const annotationsPanel = document.getElementById(
    "annotations-panel",
);

const annotationsToggle = document.getElementById(
    "annotations-toggle",
);

const annotationsList = document.getElementById(
    "annotations-panel-list",
);

const annotationsColourBySelect =
    document.getElementById(
        "annotations-colour-by",
    );

const annotationsSecondaryTypeField =
    document.getElementById(
        "annotations-secondary-type-field",
    );

const annotationsSecondaryTypeSelect =
    document.getElementById(
        "annotations-secondary-type",
    );

const annotationsPropertyField =
    document.getElementById(
        "annotations-property-field",
    );

const annotationsPropertySelect =
    document.getElementById(
        "annotations-property",
    );

const annotationsPropertyLegend =
    document.getElementById(
        "annotations-property-legend",
    );

const annotationsPropertyLegendCaption =
    document.getElementById(
        "annotations-property-legend-caption",
    );

const annotationsPropertyMin =
    document.getElementById(
        "annotations-property-min",
    );

const annotationsPropertyMax =
    document.getElementById(
        "annotations-property-max",
    );

const annotationsLinkOpacityInput =
    document.getElementById(
        "annotations-link-opacity",
    );

const annotationInspector =
    document.getElementById(
        "annotation-inspector",
    );

const annotationInspectorHeader =
    document.getElementById(
        "annotation-inspector-header",
    );

const annotationInspectorTitle =
    document.getElementById(
        "annotation-inspector-title",
    );

const annotationInspectorProperties =
    document.getElementById(
        "annotation-inspector-properties",
    );

const annotationInspectorClose =
    document.getElementById(
        "annotation-inspector-close",
    );

const annotationsSelectAllButton =
    document.getElementById(
        "annotations-select-all",
    );

const annotationsDeselectAllButton =
    document.getElementById(
        "annotations-deselect-all",
    );

const annotationsImportColoursButton =
    document.getElementById(
        "annotations-import-colours",
    );

const annotationsImportColoursInput =
    document.getElementById(
        "annotations-import-colours-file",
    );

const annotationsExportColoursButton =
    document.getElementById(
        "annotations-export-colours",
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

const annotationInspectionEnabledInput =
    document.getElementById(
        "settings-annotation-inspection",
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
    annotationsPanel === null ||
    annotationsToggle === null ||
    annotationsList === null ||
    annotationsSelectAllButton === null ||
    annotationsDeselectAllButton === null ||
    annotationsImportColoursButton === null ||
    annotationsImportColoursInput === null ||
    annotationsExportColoursButton === null ||
    annotationsColourBySelect === null ||
    annotationsPropertyField === null ||
    annotationsPropertySelect === null ||
    annotationsPropertyLegend === null ||
    annotationsPropertyLegendCaption === null ||
    annotationsPropertyMin === null ||
    annotationsPropertyMax === null ||
    annotationsLinkOpacityInput === null ||
    annotationInspector === null ||
    annotationInspectorHeader === null ||
    annotationInspectorTitle === null ||
    annotationInspectorProperties === null ||
    annotationInspectorClose === null ||
    annotationsSecondaryTypeField === null ||
    annotationsSecondaryTypeSelect === null ||
    settingsPanel === null ||
    settingsToggle === null ||
    settingsCloseButton === null ||
    annotationInspectionEnabledInput === null ||
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

let annotationInspectionRequestId = 0;

function invalidateAnnotationInspectionRequests() {
    annotationInspectionRequestId += 1;
}

function hideAnnotationInspector() {
    invalidateAnnotationInspectionRequests();
    annotationInspector.hidden = true;
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
        annotationInspectionEnabledInput,
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

        onAnnotationInspectionChange() {
            if (
                !annotationInspectionEnabledInput.checked
            ) {
                hideAnnotationInspector();
            }
        },

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
            annotationsPanelController.setOpen(false);
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
const annotationColours = new Map();
const annotationColoursByLayer = new Map();

const annotationTypesByLayer = new Map();

const annotationTypeVisibilityByLayer =
    new Map();

const annotationTypeOpacityByLayer =
    new Map();

let annotationDisplayMode = "type";
let annotationProperty = null;
let annotationSecondarySelection = null;
let annotationProperties = [];
const annotationPropertyRanges =
    new Map();

let annotationOpacityLinked = false;

function getAnnotationTypes() {
    const annotationTypes = new Set();

    for (const types of annotationTypesByLayer.values()) {
        for (const type of types) {
            annotationTypes.add(type);
        }
    }

    return [...annotationTypes];
}

function getAnnotationGroups() {
    return [...annotationTypesByLayer.entries()].map(
        ([layerName, annotationTypes]) => ({
            layerName,
            annotationTypes,
        }),
    );
}

function getFirstAnnotationSelection() {
    for (const [
        layerName,
        annotationTypes,
    ] of annotationTypesByLayer) {
        if (annotationTypes.length > 0) {
            return {
                layerName,
                annotationType:
                    annotationTypes[0],
            };
        }
    }

    return null;
}

async function getCommonAnnotationProperties() {
    const layerNames =
        [...annotationLayerNames];

    if (layerNames.length === 0) {
        return [];
    }

    const propertiesByLayer =
        await Promise.all(
            layerNames.map(
                (layerName) =>
                    getAnnotationProperties(
                        layerName,
                    ),
            ),
        );

    const commonProperties =
        propertiesByLayer[0].filter(
            (property) =>
                ![
                    "type",
                    "class",
                ].includes(
                    property.toLowerCase(),
                ) &&
                propertiesByLayer
                    .slice(1)
                    .every(
                        (properties) =>
                            properties.includes(
                                property,
                            ),
                    ),
        );

    const numericProperties = [];

    for (const property of commonProperties) {
        const valuesByLayer =
            await Promise.all(
                layerNames.map(
                    (layerName) =>
                        getAnnotationPropertyValues(
                            layerName,
                            property,
                        ),
                ),
            );

        const values =
            valuesByLayer.flat();

        const range =
            getFiniteNumberRange(
                values,
            );

        if (range !== null) {
            annotationPropertyRanges.set(
                property,
                range,
            );

            numericProperties.push(
                property,
            );
        }
    }

    return numericProperties.sort();
}

function getAnnotationPropertyRenderRange(
    property,
) {
    const range =
        annotationPropertyRanges.get(
            property,
        );

    if (range === undefined) {
        throw new Error(
            `Annotation property range is not available: ${property}`,
        );
    }

    const [
        minimum,
        maximum,
    ] = range;

    if (minimum === maximum) {
        return [
            minimum,
            minimum + 1,
        ];
    }

    return range;
}

async function setAnnotationLayerTypeMode(
    layerName,
) {
    const colours =
        annotationColoursByLayer.get(
            layerName,
        );

    if (colours === undefined) {
        throw new Error(
            `Annotation colours are not available for layer: ${layerName}`,
        );
    }

    await clearAnnotationSecondaryMapper(
        layerName,
    );

    await setAnnotationProperty(
        "type",
        layerName,
    );

    await setAnnotationPropertyRange(
        null,
        layerName,
    );

    await setTileServerAnnotationColors(
        colours,
        layerName,
    );
}

async function setAnnotationTypeMode({
    refresh = true,
} = {}) {
    await Promise.all(
        [...annotationLayerNames].map(
            (layerName) =>
                setAnnotationLayerTypeMode(
                    layerName,
                ),
        ),
    );

    annotationDisplayMode = "type";
    annotationProperty = null;
    annotationSecondarySelection = null;

    await updateAnnotationFilters({
        refresh,
    });
}

async function resetAnnotationRenderer() {
    await clearAnnotationSecondaryMapper();

    await setAnnotationProperty(
        "type",
    );

    await setAnnotationPropertyRange(
        null,
    );
}

async function setAnnotationPropertyMode(
    property,
    {
        refresh = true,
    } = {},
) {
    const range =
        getAnnotationPropertyRenderRange(
            property,
        );

    await Promise.all(
        [...annotationLayerNames].map(
            async (layerName) => {
                await clearAnnotationSecondaryMapper(
                    layerName,
                );

                await setAnnotationProperty(
                    property,
                    layerName,
                );

                await setAnnotationMapper(
                    "viridis",
                    layerName,
                );

                await setAnnotationPropertyRange(
                    range,
                    layerName,
                );
            },
        ),
    );

    annotationDisplayMode =
        "property";

    annotationProperty =
        property;

    annotationSecondarySelection =
        null;

    await updateAnnotationFilters({
        refresh,
    });
}

async function setAnnotationSecondaryMode(
    layerName,
    annotationType,
    property,
    {
        refresh = true,
    } = {},
) {
    const range =
        getAnnotationPropertyRenderRange(
            property,
        );

    await Promise.all(
        [...annotationLayerNames].map(
            (currentLayerName) =>
                setAnnotationLayerTypeMode(
                    currentLayerName,
                ),
        ),
    );

    await setAnnotationSecondaryMapper(
        annotationType,
        property,
        "viridis",
        range,
        layerName,
    );

    annotationDisplayMode = "secondary";
    annotationProperty = property;

    annotationSecondarySelection = {
        layerName,
        annotationType,
    };

    await updateAnnotationFilters({
        refresh,
    });
}

async function updateAnnotationProperties({
    refresh = true,
} = {}) {
    if (annotationLayerNames.size === 0) {
        await resetAnnotationRenderer();

        annotationProperties = [];
        annotationPropertyRanges.clear();
        annotationDisplayMode = "type";
        annotationProperty = null;
        annotationSecondarySelection = null;

        annotationsPanelController.render();
        return;
    }

    annotationProperties = [];
    annotationPropertyRanges.clear();

    annotationsPanelController.render();

    annotationProperties =
        await getCommonAnnotationProperties();

    if (annotationDisplayMode === "property") {
        if (
            annotationProperty === null ||
            !annotationProperties.includes(
                annotationProperty,
            )
        ) {
            await setAnnotationTypeMode({
                refresh,
            });
        } else {
            await setAnnotationPropertyMode(
                annotationProperty,
                {
                    refresh,
                },
            );
        }
    }

    if (annotationDisplayMode === "secondary") {
        const selection =
            annotationSecondarySelection;

        const annotationTypes =
            selection === null
                ? undefined
                : annotationTypesByLayer.get(
                    selection.layerName,
                );

        if (
            selection === null ||
            annotationTypes === undefined ||
            !annotationTypes.some(
                (annotationType) =>
                    Object.is(
                        annotationType,
                        selection.annotationType,
                    ),
            ) ||
            annotationProperty === null ||
            !annotationProperties.includes(
                annotationProperty,
            )
        ) {
            await setAnnotationTypeMode({
                refresh,
            });
        } else {
            await setAnnotationSecondaryMode(
                selection.layerName,
                selection.annotationType,
                annotationProperty,
                {
                    refresh,
                },
            );
        }
    }

    annotationsPanelController.render();
}

async function importAnnotationColours(file) {
    let config;

    try {
        config = JSON.parse(
            await file.text(),
        );
    } catch {
        throw new Error(
            "Annotation colour file is not valid JSON.",
        );
    }

    const {
        colorDict,
        layerColorDicts,
    } = parseAnnotationColourConfig(
        config,
    );

    const updates = [];

    for (const [
        layerName,
        annotationTypes,
    ] of annotationTypesByLayer) {
        const currentColours =
            annotationColoursByLayer.get(
                layerName,
            );

        if (currentColours === undefined) {
            continue;
        }

        const updatedColours =
            mergeAnnotationColourConfig(
                currentColours,
                annotationTypes,
                colorDict,
                layerColorDicts[
                    layerName
                ] ?? {},
            );

        updates.push({
            layerName,
            previousColours:
                new Map(
                    currentColours,
                ),
            updatedColours,
        });
    }

    try {
        await Promise.all(
            updates.map(
                ({
                    layerName,
                    updatedColours,
                }) =>
                    setTileServerAnnotationColors(
                        updatedColours,
                        layerName,
                    ),
            ),
        );
    } catch (error) {
        await Promise.allSettled(
            updates.map(
                ({
                    layerName,
                    previousColours,
                }) =>
                    setTileServerAnnotationColors(
                        previousColours,
                        layerName,
                    ),
            ),
        );

        throw error;
    }

    for (const {
        layerName,
        updatedColours,
    } of updates) {
        annotationColoursByLayer.set(
            layerName,
            updatedColours,
        );
    }

    refreshAnnotationLayers();
}

function exportAnnotationColours() {
    const colours =
        new Map();

    const layerColorDicts = {};

    for (const [
        layerName,
        annotationTypes,
    ] of annotationTypesByLayer) {
        const layerColours =
            annotationColoursByLayer.get(
                layerName,
            );

        if (layerColours === undefined) {
            continue;
        }

        layerColorDicts[layerName] =
            createAnnotationColourConfig(
                layerColours,
                annotationTypes,
            ).color_dict;

        for (
            const annotationType of
            annotationTypes
        ) {
            if (
                colours.has(
                    annotationType,
                )
            ) {
                continue;
            }

            const colour =
                layerColours.get(
                    annotationType,
                );

            if (colour !== undefined) {
                colours.set(
                    annotationType,
                    colour,
                );
            }
        }
    }

    const config =
        createAnnotationColourConfig(
            colours,
            getAnnotationTypes(),
        );

    config.layer_color_dicts =
        layerColorDicts;

    const json = `${JSON.stringify(
        config,
        null,
        4,
    )}\n`;

    const blob = new Blob(
        [json],
        {
            type: "application/json",
        },
    );

    const url =
        URL.createObjectURL(blob);

    const downloadLink =
        document.createElement("a");

    downloadLink.href = url;
    downloadLink.download =
        "annotation_config.json";

    document.body.appendChild(
        downloadLink,
    );

    downloadLink.click();
    downloadLink.remove();

    URL.revokeObjectURL(url);
}

function initialiseAnnotationLayerState(
    layerName,
    annotationTypes,
    colours,
) {
    const visibility = new Map(
        annotationTypes.map(
            (annotationType) => [
                annotationType,
                true,
            ],
        ),
    );

    const opacities = new Map(
        annotationTypes.map(
            (annotationType) => [
                annotationType,
                colours.get(
                    annotationType,
                )?.[3] ?? 1,
            ],
        ),
    );

    annotationColoursByLayer.set(
        layerName,
        colours,
    );

    annotationTypeVisibilityByLayer.set(
        layerName,
        visibility,
    );

    annotationTypeOpacityByLayer.set(
        layerName,
        opacities,
    );
}

function removeAnnotationLayerState(
    layerName,
) {
    annotationColoursByLayer.delete(
        layerName,
    );

    annotationTypeVisibilityByLayer.delete(
        layerName,
    );

    annotationTypeOpacityByLayer.delete(
        layerName,
    );
}

function refreshAnnotationLayers(
    excludedLayerName = null,
) {
    overlayVersion += 1;

    for (const layerName of annotationLayerNames) {
        if (layerName === excludedLayerName) {
            continue;
        }

        const overlayLayer =
            overlayLayers[layerName];

        if (overlayLayer === undefined) {
            continue;
        }

        const source = new Zoomify({
            url:
                `/tileserver/layer/${encodeURIComponent(layerName)}/` +
                `${sessionId}/zoomify/` +
                `{TileGroup}/{z}-{x}-{y}@1x.jpg?v=${overlayVersion}`,
            size:
                currentSlideInfo.slide_dimensions,
            crossOrigin: "anonymous",
            zDirection: -1,
        });

        overlayLayer.setSource(source);
    }

    map.render();
}

function isAnnotationTypeDisplayed(
    layerName,
    annotationType,
) {
    if (
        annotationDisplayMode ===
            "secondary" &&
        annotationSecondarySelection !==
            null
    ) {
        return (
            layerName ===
                annotationSecondarySelection.layerName &&
            Object.is(
                annotationType,
                annotationSecondarySelection.annotationType,
            )
        );
    }

    return (
        annotationTypeVisibilityByLayer
            .get(layerName)
            ?.get(annotationType) ??
        true
    );
}

async function updateAnnotationLayerFilter(
    layerName,
    {
        refresh = true,
    } = {},
) {
    if (!annotationLayerNames.has(layerName)) {
        return;
    }

    const annotationTypes =
        annotationTypesByLayer.get(
            layerName,
        ) ?? [];

    const visibility =
        annotationTypeVisibilityByLayer.get(
            layerName,
        );

    if (visibility === undefined) {
        throw new Error(
            `Annotation visibility is not available for layer: ${layerName}`,
        );
    }

    const displayVisibility =
        annotationDisplayMode ===
        "secondary"
            ? new Map(
                annotationTypes.map(
                    (annotationType) => [
                        annotationType,
                        isAnnotationTypeDisplayed(
                            layerName,
                            annotationType,
                        ),
                    ],
                ),
            )
            : visibility;

    await setTileServerAnnotationFilter(
        getAnnotationFilter(
            annotationTypes,
            displayVisibility,
        ),
        layerName,
    );

    if (refresh) {
        refreshAnnotationLayers();
    }
}

async function updateAnnotationFilters({
    refresh = true,
} = {}) {
    await Promise.all(
        [...annotationLayerNames].map(
            (layerName) =>
                updateAnnotationLayerFilter(
                    layerName,
                    {
                        refresh: false,
                    },
                ),
        ),
    );

    if (refresh) {
        refreshAnnotationLayers();
    }
}

async function inspectAnnotationAtCoordinate(
    coordinate,
) {
    const [
        x,
        y,
    ] = coordinate;

    const layerNames =
        [...annotationLayerNames]
            .filter(
                (layerName) =>
                    overlayLayers[
                        layerName
                    ]?.getVisible() !== false,
            )
            .sort(
                (firstLayerName, secondLayerName) =>
                    (
                        overlayLayers[
                            secondLayerName
                        ]?.getZIndex() ?? 0
                    ) -
                    (
                        overlayLayers[
                            firstLayerName
                        ]?.getZIndex() ?? 0
                    ),
            );

    for (const layerName of layerNames) {
        const properties =
            await getAnnotationAtPoint(
                layerName,
                x,
                -y,
            );

        if (
            Object.keys(
                properties,
            ).length === 0
        ) {
            continue;
        }

        if (
            properties.type !== undefined &&
            !isAnnotationTypeDisplayed(
                layerName,
                properties.type,
            )
        ) {
            continue;
        }

        return {
            layerName,
            properties,
        };
    }

    return null;
}

function formatAnnotationPropertyName(property) {
    const labels = {
        box: "Bounding box (px)",
        centroid: "Centroid (px)",
        prob: "Probability",
        type: "Type",
    };

    return labels[property] ?? property;
}

function formatAnnotationPropertyValue(
    property,
    value,
) {
    if (
        property === "prob" &&
        typeof value === "number"
    ) {
        return `${value.toFixed(4)} (${(value * 100).toFixed(2)}%)`;
    }

    if (
        property === "centroid" &&
        Array.isArray(value) &&
        value.length >= 2
    ) {
        const [
            x,
            y,
        ] = value;

        return (
            `x: ${typeof x === "number" ? x.toFixed(2) : x}, ` +
            `y: ${typeof y === "number" ? y.toFixed(2) : y}`
        );
    }

    if (
        property === "box" &&
        Array.isArray(value) &&
        value.length === 4
    ) {
        return `x: ${value[0]}–${value[2]}, y: ${value[1]}–${value[3]}`;
    }

    if (Array.isArray(value)) {
        return value.join(", ");
    }

    if (
        typeof value === "number" &&
        !Number.isInteger(value)
    ) {
        return value.toFixed(4);
    }

    if (
        value !== null &&
        typeof value === "object"
    ) {
        return JSON.stringify(value);
    }

    return String(value);
}

function positionAnnotationInspector(
    requestedLeft,
    requestedTop,
) {
    const margin = 8;

    const maxLeft = Math.max(
        margin,
        viewerApp.clientWidth -
            annotationInspector.offsetWidth -
            margin,
    );

    const maxTop = Math.max(
        margin,
        viewerApp.clientHeight -
            annotationInspector.offsetHeight -
            margin,
    );

    annotationInspector.style.left =
        `${Math.min(
            Math.max(requestedLeft, margin),
            maxLeft,
        )}px`;

    annotationInspector.style.top =
        `${Math.min(
            Math.max(requestedTop, margin),
            maxTop,
        )}px`;
}

function showAnnotationInspector(
    inspection,
    pixel,
) {
    annotationInspectorTitle.textContent =
        inspection.layerName;

    annotationInspectorProperties.replaceChildren();

    for (
        const [
            property,
            value,
        ] of Object.entries(
            inspection.properties,
        )
    ) {
        const row =
            document.createElement("div");

        row.className =
            "annotation-inspector-property";

        const name =
            document.createElement("div");

        name.className =
            "annotation-inspector-property-name";

        name.textContent =
            formatAnnotationPropertyName(
                property,
            );

        const propertyValue =
            document.createElement("div");

        propertyValue.className =
            "annotation-inspector-property-value";

        propertyValue.textContent =
            formatAnnotationPropertyValue(
                property,
                value,
            );

        row.append(
            name,
            propertyValue,
        );

        annotationInspectorProperties.append(
            row,
        );
    }

    annotationInspector.hidden = false;

    const mapRect =
        mapElement.getBoundingClientRect();

    const viewerRect =
        viewerApp.getBoundingClientRect();

    positionAnnotationInspector(
        mapRect.left -
            viewerRect.left +
            pixel[0] +
            12,
        mapRect.top -
            viewerRect.top +
            pixel[1] +
            12,
    );
}

const configuredSlides =
    await getConfiguredFiles("slide");

const configuredOverlays =
    await getConfiguredFiles("overlay");

const configuredColourDict =
    configuredOverlays.config?.color_dict ?? {};

const configuredLayerColourDicts =
    configuredOverlays.config
        ?.layer_color_dicts ?? {};

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
            layersPanelController.setOpen(false);
            annotationsPanelController.setOpen(false);
        },
    });

const annotationsPanelController =
    createAnnotationsPanelController({
        panel: annotationsPanel,
        toggle: annotationsToggle,
        list: annotationsList,
        colourBySelect:
            annotationsColourBySelect,
        secondaryTypeField:
            annotationsSecondaryTypeField,
        secondaryTypeSelect:
            annotationsSecondaryTypeSelect,
        propertyField:
            annotationsPropertyField,
        propertySelect:
            annotationsPropertySelect,
        propertyLegend:
            annotationsPropertyLegend,
        propertyLegendCaption:
            annotationsPropertyLegendCaption,
        propertyMin:
            annotationsPropertyMin,
        propertyMax:
            annotationsPropertyMax,
        linkOpacityInput:
            annotationsLinkOpacityInput,
        selectAllButton:
            annotationsSelectAllButton,
        deselectAllButton:
            annotationsDeselectAllButton,
        importButton:
            annotationsImportColoursButton,
        importInput:
            annotationsImportColoursInput,
        exportButton:
            annotationsExportColoursButton,

        getAnnotationGroups,
        getAnnotationTypes,

        getDisplayMode: () =>
            annotationDisplayMode,

        getAnnotationProperties: () =>
            annotationProperties,

        getAnnotationProperty: () =>
            annotationProperty,

        getSecondaryType: () =>
            annotationSecondarySelection,

        getPropertyRange: () =>
            annotationProperty === null
                ? null
                : annotationPropertyRanges.get(
                    annotationProperty,
                ) ?? null,

        getAnnotationColour: (
            layerName,
            annotationType,
        ) =>
            annotationColoursByLayer
                .get(layerName)
                ?.get(annotationType) ??
            [0, 0, 0, 1],

        isAnnotationTypeVisible: (
            layerName,
            annotationType,
        ) =>
            annotationTypeVisibilityByLayer
                .get(layerName)
                ?.get(annotationType) ??
            true,

        getAnnotationOpacity: (
            layerName,
            annotationType,
        ) =>
            annotationTypeOpacityByLayer
                .get(layerName)
                ?.get(annotationType) ??
            1,

        getOpacityLinked: () =>
            annotationOpacityLinked,

        async onDisplayModeChange(mode) {
            if (mode === "type") {
                await setAnnotationTypeMode();
                return;
            }

            const property =
                annotationProperty ??
                annotationProperties[0];

            if (property === undefined) {
                throw new Error(
                    "No annotation properties are available.",
                );
            }

            if (mode === "property") {
                await setAnnotationPropertyMode(
                    property,
                );

                return;
            }

            if (mode === "secondary") {
                const selection =
                    annotationSecondarySelection ??
                    getFirstAnnotationSelection();

                if (selection === null) {
                    throw new Error(
                        "No annotation classes are available.",
                    );
                }

                await setAnnotationSecondaryMode(
                    selection.layerName,
                    selection.annotationType,
                    property,
                );

                return;
            }

            throw new Error(
                `Unknown annotation display mode: ${mode}`,
            );
        },

        async onPropertyChange(property) {
            if (
                annotationDisplayMode ===
                "secondary"
            ) {
                if (
                    annotationSecondarySelection ===
                    null
                ) {
                    throw new Error(
                        "No secondary annotation class is selected.",
                    );
                }

                await setAnnotationSecondaryMode(
                    annotationSecondarySelection.layerName,
                    annotationSecondarySelection.annotationType,
                    property,
                );

                return;
            }

            await setAnnotationPropertyMode(
                property,
            );
        },

        async onSecondaryTypeChange(
            layerName,
            annotationType,
        ) {
            const property =
                annotationProperty ??
                annotationProperties[0];

            if (property === undefined) {
                throw new Error(
                    "No annotation properties are available.",
                );
            }

            await setAnnotationSecondaryMode(
                layerName,
                annotationType,
                property,
            );
        },

        async onColourChange(
            layerName,
            annotationType,
            colourValue,
        ) {
            const rgb = hexToRgb(colourValue);

            const colours =
                annotationColoursByLayer.get(
                    layerName,
                );

            if (colours === undefined) {
                throw new Error(
                    `Annotation colours are not available for layer: ${layerName}`,
                );
            }

            const currentColour =
                colours.get(annotationType);

            const opacity =
                currentColour?.[3] ?? 1;

            const updatedColours =
                new Map(colours);

            updatedColours.set(
                annotationType,
                [
                    rgb.r / 255,
                    rgb.g / 255,
                    rgb.b / 255,
                    opacity,
                ],
            );

            await setTileServerAnnotationColors(
                updatedColours,
                layerName,
            );

            annotationColoursByLayer.set(
                layerName,
                updatedColours,
            );

            refreshAnnotationLayers();
        },

        async onVisibilityChange(
            layerName,
            annotationType,
            visible,
        ) {
            const visibility =
                annotationTypeVisibilityByLayer.get(
                    layerName,
                );

            if (visibility === undefined) {
                throw new Error(
                    `Annotation visibility is not available for layer: ${layerName}`,
                );
            }

            const previousVisibility =
                visibility.get(
                    annotationType,
                ) ?? true;

            visibility.set(
                annotationType,
                visible,
            );

            try {
                await updateAnnotationLayerFilter(
                    layerName,
                );
            } catch (error) {
                visibility.set(
                    annotationType,
                    previousVisibility,
                );

                throw error;
            }
        },

        async onOpacityChange(
            layerName,
            annotationType,
            opacity,
        ) {
            const updates = [];

            const addLayerUpdate = (
                currentLayerName,
                annotationTypes,
            ) => {
                const opacities =
                    annotationTypeOpacityByLayer.get(
                        currentLayerName,
                    );

                if (opacities === undefined) {
                    throw new Error(
                        `Annotation opacity is not available for layer: ${currentLayerName}`,
                    );
                }

                const colours =
                    annotationColoursByLayer.get(
                        currentLayerName,
                    );

                if (colours === undefined) {
                    throw new Error(
                        `Annotation colours are not available for layer: ${currentLayerName}`,
                    );
                }

                const previousOpacities =
                    new Map(opacities);

                const updatedOpacities =
                    new Map(opacities);

                const updatedColours =
                    new Map(colours);

                for (
                    const currentType of
                    annotationTypes
                ) {
                    updatedOpacities.set(
                        currentType,
                        opacity,
                    );

                    const colour =
                        updatedColours.get(
                            currentType,
                        );

                    if (colour !== undefined) {
                        updatedColours.set(
                            currentType,
                            [
                                colour[0],
                                colour[1],
                                colour[2],
                                opacity,
                            ],
                        );
                    }
                }

                updates.push({
                    layerName:
                        currentLayerName,
                    previousOpacities,
                    updatedOpacities,
                    updatedColours,
                });
            };

            if (annotationOpacityLinked) {
                for (
                    const currentLayerName of
                    annotationLayerNames
                ) {
                    addLayerUpdate(
                        currentLayerName,
                        annotationTypesByLayer.get(
                            currentLayerName,
                        ) ?? [],
                    );
                }
            } else {
                addLayerUpdate(
                    layerName,
                    [annotationType],
                );
            }

            try {
                await Promise.all(
                    updates.map(
                        ({
                            layerName:
                                currentLayerName,
                            updatedOpacities,
                        }) =>
                            setTileServerAnnotationOpacities(
                                updatedOpacities,
                                currentLayerName,
                            ),
                    ),
                );
            } catch (error) {
                await Promise.allSettled(
                    updates.map(
                        ({
                            layerName:
                                currentLayerName,
                            previousOpacities,
                        }) =>
                            setTileServerAnnotationOpacities(
                                previousOpacities,
                                currentLayerName,
                            ),
                    ),
                );

                throw error;
            }

            for (
                const {
                    layerName:
                        currentLayerName,
                    updatedOpacities,
                    updatedColours,
                } of updates
            ) {
                annotationTypeOpacityByLayer.set(
                    currentLayerName,
                    updatedOpacities,
                );

                annotationColoursByLayer.set(
                    currentLayerName,
                    updatedColours,
                );
            }

            refreshAnnotationLayers();
        },

        async onOpacityLinkChange(linked) {
            annotationOpacityLinked = linked;
        },

        async onImport(file) {
            await importAnnotationColours(
                file,
            );
        },

        onExport() {
            exportAnnotationColours();
        },

        onOpen() {
            filesPanelController.setOpen(false);
            layersPanelController.setOpen(false);
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

const map = new OlMap({
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

map.on("singleclick", async (event) => {
    if (
        !annotationInspectionEnabledInput.checked ||
        annotationLayerNames.size === 0
    ) {
        return;
    }

    annotationInspectionRequestId += 1;

    const requestId =
        annotationInspectionRequestId;

    try {
        const inspection =
            await inspectAnnotationAtCoordinate(
                event.coordinate,
            );

        if (
            requestId !==
                annotationInspectionRequestId ||
            !annotationInspectionEnabledInput.checked ||
            annotationLayerNames.size === 0
        ) {
            return;
        }

        if (inspection === null) {
            return;
        }

        showAnnotationInspector(
            inspection,
            event.pixel,
        );
    } catch (error) {
        if (
            requestId !==
            annotationInspectionRequestId
        ) {
            return;
        }

        console.error(
            "Failed to inspect annotation.",
            error,
        );
    }
});

annotationInspectorClose.addEventListener(
    "click",
    hideAnnotationInspector,
);

let annotationInspectorDragging = false;
let annotationInspectorDragOffsetX = 0;
let annotationInspectorDragOffsetY = 0;

annotationInspectorHeader.addEventListener(
    "pointerdown",
    (event) => {
        if (
            event.target.closest("button") !== null
        ) {
            return;
        }

        event.preventDefault();

        const inspectorRect =
            annotationInspector.getBoundingClientRect();

        annotationInspectorDragging = true;

        annotationInspectorDragOffsetX =
            event.clientX -
            inspectorRect.left;

        annotationInspectorDragOffsetY =
            event.clientY -
            inspectorRect.top;

        annotationInspectorHeader.setPointerCapture(
            event.pointerId,
        );

        annotationInspector.classList.add(
            "dragging",
        );
    },
);

annotationInspectorHeader.addEventListener(
    "pointermove",
    (event) => {
        if (!annotationInspectorDragging) {
            return;
        }

        const viewerRect =
            viewerApp.getBoundingClientRect();

        positionAnnotationInspector(
            event.clientX -
                viewerRect.left -
                annotationInspectorDragOffsetX,
            event.clientY -
                viewerRect.top -
                annotationInspectorDragOffsetY,
        );
    },
);

function stopAnnotationInspectorDrag() {
    annotationInspectorDragging = false;

    annotationInspector.classList.remove(
        "dragging",
    );
}

annotationInspectorHeader.addEventListener(
    "pointerup",
    stopAnnotationInspectorDrag,
);

annotationInspectorHeader.addEventListener(
    "pointercancel",
    stopAnnotationInspectorDrag,
);

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
    annotationTypesByLayer.clear();

    annotationColoursByLayer.clear();
    annotationTypeVisibilityByLayer.clear();
    annotationTypeOpacityByLayer.clear();

    hideAnnotationInspector();

    annotationDisplayMode = "type";
    annotationProperty = null;
    annotationSecondarySelection = null;
    annotationProperties = [];
    annotationPropertyRanges.clear();

    annotationsPanelController.render();
    layersPanelController.render();
    filesPanelController.updateActionState();
}

async function clearOverlays() {
    await clearTileServerOverlays();

    clearOverlayLayers();

    await resetAnnotationRenderer();
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

    await resetAnnotationRenderer();

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

    await resetAnnotationRenderer();

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
    const wasAnnotation =
        annotationLayerNames.has(layerName);

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

        const annotationTypes =
            [...new Set(result)];

        annotationTypesByLayer.set(
            layerName,
            annotationTypes,
        );

        const layerColours =
            new Map();

        const configuredColours = {
            ...configuredColourDict,
            ...(
                configuredLayerColourDicts[
                    layerName
                ] ?? {}
            ),
        };

        await assignAnnotationColours(
            layerColours,
            annotationTypes,
            getTileServerAnnotationColors,
            configuredColours,
        );

        for (const [
            annotationType,
            colour,
        ] of layerColours) {
            if (
                !annotationColours.has(
                    annotationType,
                )
            ) {
                annotationColours.set(
                    annotationType,
                    colour,
                );
            }
        }

        initialiseAnnotationLayerState(
            layerName,
            annotationTypes,
            layerColours,
        );

        await setTileServerAnnotationOpacities(
            annotationTypeOpacityByLayer.get(
                layerName,
            ),
            layerName,
        );

        if (annotationDisplayMode === "type") {
            await setTileServerAnnotationColors(
                annotationColoursByLayer.get(
                    layerName,
                ),
                layerName,
            );
        }
    } else if (wasAnnotation) {
        annotationLayerNames.delete(layerName);
        annotationTypesByLayer.delete(layerName);

        removeAnnotationLayerState(
            layerName,
        );
    }

    if (isAnnotation || wasAnnotation) {
        await updateAnnotationProperties({
            refresh: false,
        });

        if (isAnnotation) {
            await updateAnnotationLayerFilter(
                layerName,
                {
                    refresh: false,
                },
            );
        }

        refreshAnnotationLayers(
            layerName,
        );
    } else {
        overlayVersion += 1;
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
    const wasAnnotation =
        annotationLayerNames.has(layerName);

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
    annotationTypesByLayer.delete(layerName);

    removeAnnotationLayerState(
        layerName,
    );

    delete overlayLayers[layerName];

    if (wasAnnotation) {
        hideAnnotationInspector();

        await updateAnnotationProperties({
            refresh: false,
        });

        refreshAnnotationLayers();
    }

    layersPanelController.render();
    filesPanelController.updateActionState();
}

async function setAnnotationColors(colorMap) {
    if (annotationLayerNames.size === 0) {
        throw new Error("No annotation overlay is loaded.");
    }

    const entries =
        colorMap instanceof Map
            ? [...colorMap.entries()]
            : Object.entries(colorMap);

    const updatedColours =
        new Map(entries);

    await setTileServerAnnotationColors(
        updatedColours,
    );

    annotationColours.clear();

    for (const [type, colour] of updatedColours) {
        annotationColours.set(
            type,
            colour,
        );
    }

    refreshAnnotationLayers();
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
