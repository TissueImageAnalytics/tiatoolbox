import {
    beforeEach,
    describe,
    expect,
    it,
    vi,
} from "vitest";

import {
    createAnnotationsPanelController,
} from "../../../tiatoolbox/visualization/openlayers/src/panels/annotations.js";

function createHarness({
    annotationGroups = [],
    displayMode = "type",
    annotationProperties = ["prob"],
    annotationProperty = null,
    secondarySelection = null,
    onDisplayModeChange = vi.fn(async () => {}),
    onPropertyChange = vi.fn(async () => {}),
    onSecondaryTypeChange = vi.fn(async () => {}),
    opacityLinked = false,
    onOpacityLinkChange = vi.fn(async () => {}),
    onImport = vi.fn(async () => {}),
    propertyRange = [0.2, 0.8],
} = {}) {
    document.body.innerHTML = `
        <button id="toggle" type="button"></button>

        <aside id="panel" class="hidden">
            <label class="annotations-panel-display-field">
                <select id="colour-by">
                    <option value="type">Class</option>
                    <option value="property">Property</option>
                    <option value="secondary">Class + Property</option>
                </select>
            </label>

            <label id="secondary-type-field" hidden>
                <select id="secondary-type"></select>
            </label>

            <label id="property-field" hidden>
                <select id="property"></select>
            </label>

            <div id="property-legend" hidden>
                <span id="property-legend-caption"></span>
                <span id="property-min"></span>
                <span id="property-max"></span>
            </div>

            <label class="annotations-panel-link-opacity">
                <input
                    id="link-opacity"
                    type="checkbox"
                >
                Link opacity
            </label>

            <div id="list"></div>

            <button
                id="select-all"
                type="button"
                disabled
            >
                Select all
            </button>

            <button
                id="deselect-all"
                type="button"
                disabled
            >
                Deselect all
            </button>

            <button
                id="import"
                type="button"
                disabled
            >
                Import colours
            </button>

            <input
                id="import-file"
                type="file"
            >

            <button
                id="export"
                type="button"
                disabled
            >
                Export colours
            </button>
        </aside>
    `;

    const panel =
        document.getElementById("panel");

    const toggle =
        document.getElementById("toggle");

    const list =
        document.getElementById("list");

    const colourBySelect =
        document.getElementById("colour-by");

    const secondaryTypeField =
        document.getElementById(
            "secondary-type-field",
        );

    const secondaryTypeSelect =
        document.getElementById(
            "secondary-type",
        );

    const propertyField =
        document.getElementById("property-field");

    const propertySelect =
        document.getElementById("property");

    const propertyLegend =
        document.getElementById(
            "property-legend",
        );

    const propertyLegendCaption =
        document.getElementById(
            "property-legend-caption",
        );

    const propertyMin =
        document.getElementById(
            "property-min",
        );

    const propertyMax =
        document.getElementById(
            "property-max",
        );

    const linkOpacityInput =
        document.getElementById(
            "link-opacity",
        );

    const selectAllButton =
        document.getElementById("select-all");

    const deselectAllButton =
        document.getElementById("deselect-all");

    const importButton =
        document.getElementById(
            "import",
        );

    const importInput =
        document.getElementById(
            "import-file",
        );

    const exportButton =
        document.getElementById("export");

    const colours = new Map([
        [
            "Tumour",
            [1, 0.5, 0, 1],
        ],
        [
            "Stroma",
            [0, 1, 1, 1],
        ],
        [
            "Inflammatory",
            [0, 1, 0, 1],
        ],
        [
            0,
            [1, 0, 0, 1],
        ],
        [
            1,
            [0, 0, 1, 1],
        ],
    ]);

    const visibility = new Map([
        ["Tumour", true],
        ["Stroma", false],
        ["Inflammatory", true],
    ]);

    const visibilityByLayer =
        new Map(
            annotationGroups.map(
                (group) => [
                    group.layerName,
                    new Map(
                        group.annotationTypes.map(
                            (annotationType) => [
                                annotationType,
                                visibility.get(
                                    annotationType,
                                ) ?? true,
                            ],
                        ),
                    ),
                ],
            ),
        );

    const opacities = new Map([
        ["Tumour", 1],
        ["Stroma", 0.5],
        ["Inflammatory", 0.75],
    ]);

    const onColourChange =
        vi.fn(async () => {});

    const onVisibilityChange =
        vi.fn(
            async (
                layerName,
                annotationType,
                visible,
            ) => {
                visibilityByLayer
                    .get(layerName)
                    ?.set(
                        annotationType,
                        visible,
                    );
            },
        );

    const onOpacityChange =
        vi.fn(async () => {});

    const onSetAllVisibility =
        vi.fn(async () => {});

    const onExport =
        vi.fn();

    const onOpen =
        vi.fn();

    const controller =
        createAnnotationsPanelController({
            panel,
            toggle,
            list,
            colourBySelect,
            secondaryTypeField,
            secondaryTypeSelect,
            propertyField,
            propertySelect,
            propertyLegend,
            propertyLegendCaption,
            propertyMin,
            propertyMax,
            linkOpacityInput,
            selectAllButton,
            deselectAllButton,
            importButton,
            importInput,
            exportButton,

            getAnnotationGroups: () =>
                annotationGroups,

            getAnnotationTypes: () => [
                ...new Set(
                    annotationGroups.flatMap(
                        (group) =>
                            group.annotationTypes,
                    ),
                ),
            ],

            getDisplayMode: () =>
                displayMode,

            getAnnotationProperties: () =>
                annotationProperties,

            getAnnotationProperty: () =>
                annotationProperty,

            getSecondaryType: () =>
                secondarySelection,

            getPropertyRange: () =>
                propertyRange,

            getAnnotationColour:
                (
                    _layerName,
                    annotationType,
                ) =>
                    colours.get(
                        annotationType,
                    ),

            isAnnotationTypeVisible:
                (
                    layerName,
                    annotationType,
                ) =>
                    visibilityByLayer
                        .get(layerName)
                        ?.get(annotationType) ??
                    true,

            getAnnotationOpacity:
                (
                    _layerName,
                    annotationType,
                ) =>
                    opacities.get(
                        annotationType,
                    ) ?? 1,

            getOpacityLinked: () =>
                opacityLinked,

            onColourChange,
            onVisibilityChange,
            onOpacityChange,
            onOpacityLinkChange,
            onDisplayModeChange,
            onPropertyChange,
            onSecondaryTypeChange,
            onSetAllVisibility,
            onImport,
            onExport,
            onOpen,
        });

    return {
        controller,
        panel,
        toggle,
        list,
        colourBySelect,
        secondaryTypeField,
        secondaryTypeSelect,
        propertyField,
        propertySelect,
        propertyLegend,
        propertyLegendCaption,
        propertyMin,
        propertyMax,
        linkOpacityInput,
        selectAllButton,
        deselectAllButton,
        importButton,
        importInput,
        exportButton,
        colours,
        visibility,
        opacities,
        onColourChange,
        onVisibilityChange,
        onOpacityChange,
        onOpacityLinkChange,
        onDisplayModeChange,
        onPropertyChange,
        onSecondaryTypeChange,
        onSetAllVisibility,
        onImport,
        onExport,
        onOpen,
    };
}

async function flushActions() {
    await Promise.resolve();
    await Promise.resolve();
}

describe("createAnnotationsPanelController", () => {
    beforeEach(() => {
        document.body.innerHTML = "";
    });

    it("renders the empty state and disables actions", () => {
        // Test the empty annotation state and disabled panel actions.
        const {
            controller,
            list,
            colourBySelect,
            selectAllButton,
            deselectAllButton,
            importButton,
            exportButton,
            linkOpacityInput,
        } = createHarness();

        controller.render();

        expect(
            colourBySelect.disabled,
        ).toBe(true);

        expect(
            colourBySelect
                .closest(
                    ".annotations-panel-display-field",
                )
                ?.classList.contains(
                    "disabled",
                ),
        ).toBe(true);

        expect(
            list.querySelector(
                ".annotations-panel-empty",
            )?.textContent,
        ).toBe("No annotations loaded");

        expect(selectAllButton.disabled).toBe(true);
        expect(deselectAllButton.disabled).toBe(true);
        expect(importButton.disabled).toBe(true);
        expect(exportButton.disabled).toBe(true);

        expect(
            linkOpacityInput.disabled,
        ).toBe(true);
    });

    it("renders annotation groups and their state", () => {
        // Test rendering annotation groups with their current control state.
        const {
            controller,
            list,
            selectAllButton,
            deselectAllButton,
            importButton,
            exportButton,
            colours,
        } = createHarness({
            annotationGroups: [
                {
                    layerName:
                        "semantic_segmentation",
                    annotationTypes: [
                        "Tumour",
                        "Stroma",
                    ],
                },
            ],
        });

        colours.set(
            "Tumour",
            "#ff8000",
        );

        controller.render();

        const groupTitle =
            list.querySelector(
                ".annotations-panel-group-title",
            );

        expect(groupTitle?.textContent).toBe(
            "semantic_segmentation",
        );

        const items =
            list.querySelectorAll(
                ".annotations-panel-item",
            );

        expect(items).toHaveLength(2);

        const names =
            list.querySelectorAll(
                ".annotations-panel-name",
            );

        expect(
            [...names].map(
                (element) =>
                    element.textContent,
            ),
        ).toEqual([
            "Tumour",
            "Stroma",
        ]);

        const visibilityInputs =
            list.querySelectorAll(
                ".annotations-panel-visibility",
            );

        expect(
            visibilityInputs[0].checked,
        ).toBe(true);

        expect(
            visibilityInputs[1].checked,
        ).toBe(false);

        const colourInputs =
            list.querySelectorAll(
                ".annotations-panel-colour",
            );

        expect(
            colourInputs[0].value,
        ).toBe("#ff8000");

        const sliders =
            list.querySelectorAll(
                ".annotations-panel-slider",
            );

        expect(sliders[0].value).toBe("1");
        expect(sliders[1].value).toBe("0.5");

        expect(selectAllButton.disabled).toBe(false);
        expect(deselectAllButton.disabled).toBe(false);
        expect(importButton.disabled).toBe(false);
        expect(exportButton.disabled).toBe(false);
    });

    it("renders annotation display controls", () => {
        // Test Class mode is shown by default.
        const {
            controller,
            colourBySelect,
            propertyField,
        } = createHarness({
            annotationGroups: [
                {
                    layerName: "annotations",
                    annotationTypes: [
                        "Tumour",
                    ],
                },
            ],
        });

        controller.render();

        expect(
            colourBySelect.value,
        ).toBe("type");

        expect(
            colourBySelect.disabled,
        ).toBe(false);

        expect(
            colourBySelect
                .closest(
                    ".annotations-panel-display-field",
                )
                ?.classList.contains(
                    "disabled",
                ),
        ).toBe(false);

        expect(
            propertyField.hidden,
        ).toBe(true);
    });

    it("renders property mode", () => {
        // Test Property mode shows available properties.
        const {
            controller,
            list,
            propertyField,
            propertySelect,
            propertyLegend,
            propertyLegendCaption,
            propertyMin,
            propertyMax,
            importButton,
            exportButton,
        } = createHarness({
            annotationGroups: [
                {
                    layerName: "annotations",
                    annotationTypes: [
                        "Tumour",
                    ],
                },
            ],
            displayMode: "property",
            annotationProperty: "prob",
        });

        controller.render();

        expect(
            propertyField.hidden,
        ).toBe(false);

        expect(
            [...propertySelect.options].map(
                (option) => option.value,
            ),
        ).toEqual([
            "prob",
        ]);

        expect(
            propertySelect.value,
        ).toBe("prob");

        expect(
            propertyLegend.hidden,
        ).toBe(false);

        expect(
            propertyLegendCaption.textContent,
        ).toBe(
            "prob values · low → high",
        );

        expect(
            propertyMin.textContent,
        ).toBe("0.2");

        expect(
            propertyMax.textContent,
        ).toBe("0.8");

        expect(
            importButton.disabled,
        ).toBe(true);

        expect(
            exportButton.disabled,
        ).toBe(true);

        expect(
            list.querySelector(
                ".annotations-panel-colour",
            ).disabled,
        ).toBe(true);

        expect(
            list.querySelector(
                ".annotations-panel-slider",
            ).disabled,
        ).toBe(false);
    });

    it("renders class and property mode", () => {
        // Test one class can use continuous property colouring.
        const {
            controller,
            list,
            secondaryTypeField,
            secondaryTypeSelect,
            propertyField,
            propertySelect,
            propertyLegend,
            propertyLegendCaption,
            importButton,
            exportButton,
            selectAllButton,
            deselectAllButton,
        } = createHarness({
            annotationGroups: [
                {
                    layerName: "annotations",
                    annotationTypes: [
                        "Tumour",
                        "Stroma",
                    ],
                },
            ],
            displayMode: "secondary",
            annotationProperty: "prob",
            secondarySelection: {
                layerName: "annotations",
                annotationType: "Stroma",
            },
        });

        controller.render();

        expect(
            secondaryTypeField.hidden,
        ).toBe(false);

        expect(
            propertyField.hidden,
        ).toBe(false);

        expect(
            [...secondaryTypeSelect.options].map(
                (option) =>
                    option.textContent,
            ),
        ).toEqual([
            "annotations · Tumour",
            "annotations · Stroma",
        ]);

        expect(
            secondaryTypeSelect.value,
        ).toBe(
            JSON.stringify([
                "annotations",
                "Stroma",
            ]),
        );

        expect(
            propertySelect.value,
        ).toBe("prob");

        expect(
            propertyLegend.hidden,
        ).toBe(false);

        expect(
            propertyLegendCaption.textContent,
        ).toBe(
            "annotations · Stroma · prob values · low → high",
        );

        const visibilityInputs =
            list.querySelectorAll(
                ".annotations-panel-visibility",
            );

        expect(
            visibilityInputs[0].checked,
        ).toBe(false);

        expect(
            visibilityInputs[1].checked,
        ).toBe(true);

        expect(
            visibilityInputs[0].disabled,
        ).toBe(true);

        expect(
            visibilityInputs[1].disabled,
        ).toBe(true);

        expect(
            selectAllButton.disabled,
        ).toBe(true);

        expect(
            deselectAllButton.disabled,
        ).toBe(true);

        const colours =
            list.querySelectorAll(
                ".annotations-panel-colour",
            );

        expect(
            colours[0].disabled,
        ).toBe(true);

        expect(
            colours[1].disabled,
        ).toBe(true);

        const sliders =
            list.querySelectorAll(
                ".annotations-panel-slider",
            );

        expect(
            sliders[0].disabled,
        ).toBe(true);

        expect(
            sliders[1].disabled,
        ).toBe(false);

        expect(
            importButton.disabled,
        ).toBe(false);

        expect(
            exportButton.disabled,
        ).toBe(false);
    });

    it("renders repeated annotation types in separate groups", () => {
        // Test repeated annotation types are shown in each annotation group.
        const {
            controller,
            list,
        } = createHarness({
            annotationGroups: [
                {
                    layerName:
                        "semantic_segmentation",
                    annotationTypes: [
                        "Inflammatory",
                    ],
                },
                {
                    layerName:
                        "nucleus_detection",
                    annotationTypes: [
                        "Inflammatory",
                    ],
                },
            ],
        });

        controller.render();

        const titles =
            list.querySelectorAll(
                ".annotations-panel-group-title",
            );

        expect(
            [...titles].map(
                (element) =>
                    element.textContent,
            ),
        ).toEqual([
            "semantic_segmentation",
            "nucleus_detection",
        ]);

        const names =
            list.querySelectorAll(
                ".annotations-panel-name",
            );

        expect(
            [...names].map(
                (element) =>
                    element.textContent,
            ),
        ).toEqual([
            "Inflammatory",
            "Inflammatory",
        ]);
    });

    it("updates visibility and rerenders repeated types", async () => {
        // Test changing visibility updates repeated annotation type controls.
        const {
            controller,
            list,
            onVisibilityChange,
        } = createHarness({
            annotationGroups: [
                {
                    layerName: "first",
                    annotationTypes: [
                        "Inflammatory",
                    ],
                },
                {
                    layerName: "second",
                    annotationTypes: [
                        "Inflammatory",
                    ],
                },
            ],
        });

        controller.render();

        const visibilityInputs =
            list.querySelectorAll(
                ".annotations-panel-visibility",
            );

        visibilityInputs[0].checked = false;

        visibilityInputs[0].dispatchEvent(
            new Event(
                "change",
                {
                    bubbles: true,
                },
            ),
        );

        await flushActions();

        expect(
            onVisibilityChange,
        ).toHaveBeenCalledExactlyOnceWith(
            "first",
            "Inflammatory",
            false,
        );

        const rerenderedInputs =
            list.querySelectorAll(
                ".annotations-panel-visibility",
            );

        expect(
            rerenderedInputs[0].checked,
        ).toBe(false);

        expect(
            rerenderedInputs[1].checked,
        ).toBe(true);
    });

    it("calls the colour change callback", async () => {
        // Test changing an annotation colour calls the colour callback.
        const {
            controller,
            list,
            onColourChange,
        } = createHarness({
            annotationGroups: [
                {
                    layerName: "annotations",
                    annotationTypes: [
                        "Tumour",
                    ],
                },
            ],
        });

        controller.render();

        const colourInput =
            list.querySelector(
                ".annotations-panel-colour",
            );

        colourInput.value = "#123456";

        colourInput.dispatchEvent(
            new Event(
                "change",
                {
                    bubbles: true,
                },
            ),
        );

        await flushActions();

        expect(
            onColourChange,
        ).toHaveBeenCalledExactlyOnceWith(
            "annotations",
            "Tumour",
            "#123456",
        );
    });

    it("updates the opacity label without sending an action on input", () => {
        // Test moving the opacity slider only updates its displayed value.
        const {
            controller,
            list,
            onOpacityChange,
        } = createHarness({
            annotationGroups: [
                {
                    layerName: "annotations",
                    annotationTypes: [
                        "Tumour",
                    ],
                },
            ],
        });

        controller.render();

        const slider =
            list.querySelector(
                ".annotations-panel-slider",
            );

        slider.value = "0.35";

        slider.dispatchEvent(
            new Event(
                "input",
                {
                    bubbles: true,
                },
            ),
        );

        expect(
            list.querySelector(
                ".annotations-panel-value",
            )?.textContent,
        ).toBe("35%");

        expect(
            onOpacityChange,
        ).not.toHaveBeenCalled();
    });

    it("calls the opacity callback on change", async () => {
        // Test committing an opacity change calls the opacity callback.
        const {
            controller,
            list,
            onOpacityChange,
        } = createHarness({
            annotationGroups: [
                {
                    layerName: "annotations",
                    annotationTypes: [
                        "Tumour",
                    ],
                },
            ],
        });

        controller.render();

        const slider =
            list.querySelector(
                ".annotations-panel-slider",
            );

        slider.value = "0.4";

        slider.dispatchEvent(
            new Event(
                "change",
                {
                    bubbles: true,
                },
            ),
        );

        await flushActions();

        expect(
            onOpacityChange,
        ).toHaveBeenCalledExactlyOnceWith(
            "annotations",
            "Tumour",
            0.4,
        );
    });

    it("renders and changes linked opacity", async () => {
        // Test the Link opacity control reflects state and reports changes.
        const {
            controller,
            linkOpacityInput,
            onOpacityLinkChange,
        } = createHarness({
            annotationGroups: [
                {
                    layerName: "annotations",
                    annotationTypes: [
                        "Tumour",
                    ],
                },
            ],
            opacityLinked: true,
        });

        controller.render();

        expect(
            linkOpacityInput.checked,
        ).toBe(true);

        expect(
            linkOpacityInput.disabled,
        ).toBe(false);

        linkOpacityInput.checked = false;

        linkOpacityInput.dispatchEvent(
            new Event(
                "change",
                {
                    bubbles: true,
                },
            ),
        );

        await flushActions();

        expect(
            onOpacityLinkChange,
        ).toHaveBeenCalledExactlyOnceWith(
            false,
        );
    });

    it("previews linked opacity across all sliders", () => {
        // Test linked opacity updates all displayed slider values before commit.
        const {
            controller,
            list,
            onOpacityChange,
        } = createHarness({
            annotationGroups: [
                {
                    layerName: "annotations",
                    annotationTypes: [
                        "Tumour",
                        "Stroma",
                    ],
                },
            ],
            opacityLinked: true,
        });

        controller.render();

        const sliders =
            list.querySelectorAll(
                ".annotations-panel-slider",
            );

        sliders[0].value = "0.35";

        sliders[0].dispatchEvent(
            new Event(
                "input",
                {
                    bubbles: true,
                },
            ),
        );

        expect(
            [...sliders].map(
                (slider) =>
                    slider.value,
            ),
        ).toEqual([
            "0.35",
            "0.35",
        ]);

        expect(
            [
                ...list.querySelectorAll(
                    ".annotations-panel-value",
                ),
            ].map(
                (value) =>
                    value.textContent,
            ),
        ).toEqual([
            "35%",
            "35%",
        ]);

        expect(
            onOpacityChange,
        ).not.toHaveBeenCalled();
    });

    it("changes annotation display mode", async () => {
        // Test changing the annotation display mode.
        const {
            controller,
            colourBySelect,
            onDisplayModeChange,
        } = createHarness({
            annotationGroups: [
                {
                    layerName: "annotations",
                    annotationTypes: [
                        "Tumour",
                    ],
                },
            ],
        });

        controller.render();

        colourBySelect.value =
            "property";

        colourBySelect.dispatchEvent(
            new Event(
                "change",
                {
                    bubbles: true,
                },
            ),
        );

        await flushActions();

        expect(
            onDisplayModeChange,
        ).toHaveBeenCalledExactlyOnceWith(
            "property",
        );
    });

    it("changes annotation property", async () => {
        // Test changing the property used to colour annotations.
        const {
            controller,
            propertySelect,
            onPropertyChange,
        } = createHarness({
            annotationGroups: [
                {
                    layerName: "annotations",
                    annotationTypes: [
                        "Tumour",
                    ],
                },
            ],
            displayMode: "property",
            annotationProperties: [
                "prob",
                "score",
            ],
            annotationProperty: "prob",
        });

        controller.render();

        propertySelect.value =
            "score";

        propertySelect.dispatchEvent(
            new Event(
                "change",
                {
                    bubbles: true,
                },
            ),
        );

        await flushActions();

        expect(
            onPropertyChange,
        ).toHaveBeenCalledExactlyOnceWith(
            "score",
        );
    });

    it("changes the secondary annotation class", async () => {
        // Test secondary class selection preserves the annotation type value.
        const {
            controller,
            secondaryTypeSelect,
            onSecondaryTypeChange,
        } = createHarness({
            annotationGroups: [
                {
                    layerName: "annotations",
                    annotationTypes: [
                        0,
                        1,
                    ],
                },
            ],
            displayMode: "secondary",
            annotationProperty: "prob",
            secondarySelection: {
                layerName: "annotations",
                annotationType: 0,
            },
        });

        controller.render();

        secondaryTypeSelect.value =
            JSON.stringify([
                "annotations",
                1,
            ]);

        secondaryTypeSelect.dispatchEvent(
            new Event(
                "change",
                {
                    bubbles: true,
                },
            ),
        );

        await flushActions();

        expect(
            onSecondaryTypeChange,
        ).toHaveBeenCalledExactlyOnceWith(
            "annotations",
            1,
        );
    });

    it("calls the bulk visibility callbacks", async () => {
        // Test Select all and Deselect all call the bulk visibility callback.
        const {
            controller,
            selectAllButton,
            deselectAllButton,
            onSetAllVisibility,
        } = createHarness({
            annotationGroups: [
                {
                    layerName: "annotations",
                    annotationTypes: [
                        "Tumour",
                    ],
                },
            ],
        });

        controller.render();

        selectAllButton.click();

        await flushActions();

        expect(
            onSetAllVisibility,
        ).toHaveBeenCalledWith(true);

        deselectAllButton.click();

        await flushActions();

        expect(
            onSetAllVisibility,
        ).toHaveBeenCalledWith(false);

        expect(
            onSetAllVisibility,
        ).toHaveBeenCalledTimes(2);
    });

    it("opens the colour import file picker", () => {
        // Test Import colours opens the file input.
        const {
            controller,
            importButton,
            importInput,
        } = createHarness({
            annotationGroups: [
                {
                    layerName: "annotations",
                    annotationTypes: [
                        "Tumour",
                    ],
                },
            ],
        });

        const clickSpy =
            vi.spyOn(
                importInput,
                "click",
            );

        controller.render();

        importButton.click();

        expect(
            clickSpy,
        ).toHaveBeenCalledOnce();
    });

    it("ignores an empty colour file selection", () => {
        const {
            controller,
            importInput,
            onImport,
        } = createHarness({
            annotationGroups: [
                {
                    layerName: "annotations",
                    annotationTypes: [
                        "Tumour",
                    ],
                },
            ],
        });

        controller.render();

        importInput.dispatchEvent(
            new Event(
                "change",
                {
                    bubbles: true,
                },
            ),
        );

        expect(
            onImport,
        ).not.toHaveBeenCalled();
    });

    it("imports the selected colour file", async () => {
        // Test selecting a file calls the import callback.
        const {
            controller,
            importInput,
            onImport,
        } = createHarness({
            annotationGroups: [
                {
                    layerName: "annotations",
                    annotationTypes: [
                        "Tumour",
                    ],
                },
            ],
        });

        const file = {
            name:
                "annotation_config.json",
        };

        Object.defineProperty(
            importInput,
            "files",
            {
                configurable: true,
                value: [
                    file,
                ],
            },
        );

        controller.render();

        importInput.dispatchEvent(
            new Event(
                "change",
                {
                    bubbles: true,
                },
            ),
        );

        await flushActions();

        expect(
            onImport,
        ).toHaveBeenCalledExactlyOnceWith(
            file,
        );

        expect(
            importInput.value,
        ).toBe("");
    });

    it("exports annotation colours", () => {
        // Test exporting annotation colours calls the export callback.
        const {
            controller,
            exportButton,
            onExport,
        } = createHarness({
            annotationGroups: [
                {
                    layerName: "annotations",
                    annotationTypes: [
                        "Tumour",
                    ],
                },
            ],
        });

        controller.render();

        exportButton.click();

        expect(
            onExport,
        ).toHaveBeenCalledOnce();
    });

    it("opens and closes the annotations panel", () => {
        // Test the Annotations button opens and closes the panel.
        const {
            panel,
            toggle,
            onOpen,
        } = createHarness();

        expect(
            panel.classList.contains("hidden"),
        ).toBe(true);

        toggle.click();

        expect(
            panel.classList.contains("hidden"),
        ).toBe(false);

        expect(
            toggle.classList.contains("active"),
        ).toBe(true);

        expect(onOpen).toHaveBeenCalledOnce();

        toggle.click();

        expect(
            panel.classList.contains("hidden"),
        ).toBe(true);

        expect(
            toggle.classList.contains("active"),
        ).toBe(false);

        expect(onOpen).toHaveBeenCalledOnce();
    });

    it("logs failed actions and rerenders", async () => {
        // Test failed annotation actions are logged and rerendered.
        const consoleError =
            vi.spyOn(
                console,
                "error",
            ).mockImplementation(
                () => {},
            );

        const {
            controller,
            list,
            onVisibilityChange,
        } = createHarness({
            annotationGroups: [
                {
                    layerName: "annotations",
                    annotationTypes: [
                        "Tumour",
                    ],
                },
            ],
        });

        onVisibilityChange.mockRejectedValueOnce(
            new Error("Update failed"),
        );

        controller.render();

        const visibility =
            list.querySelector(
                ".annotations-panel-visibility",
            );

        visibility.checked = false;

        visibility.dispatchEvent(
            new Event(
                "change",
                {
                    bubbles: true,
                },
            ),
        );

        await flushActions();

        expect(consoleError).toHaveBeenCalledOnce();

        expect(
            list.querySelectorAll(
                ".annotations-panel-item",
            ),
        ).toHaveLength(1);

        consoleError.mockRestore();
    });
});
