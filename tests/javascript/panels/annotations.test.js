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
    secondaryType = null,
    onDisplayModeChange = vi.fn(async () => {}),
    onPropertyChange = vi.fn(async () => {}),
    onSecondaryTypeChange = vi.fn(async () => {}),
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

            <div id="list"></div>

            <button
                id="show-all"
                type="button"
                disabled
            >
                Show all
            </button>

            <button
                id="hide-all"
                type="button"
                disabled
            >
                Hide all
            </button>

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

    const showAllButton =
        document.getElementById("show-all");

    const hideAllButton =
        document.getElementById("hide-all");

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
                annotationType,
                visible,
            ) => {
                visibility.set(
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
            showAllButton,
            hideAllButton,
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
                secondaryType,

            getPropertyRange: () =>
                propertyRange,

            getAnnotationColour:
                (annotationType) =>
                    colours.get(
                        annotationType,
                    ),

            isAnnotationTypeVisible:
                (annotationType) =>
                    visibility.get(
                        annotationType,
                    ) ?? true,

            getAnnotationOpacity:
                (annotationType) =>
                    opacities.get(
                        annotationType,
                    ) ?? 1,

            onColourChange,
            onVisibilityChange,
            onOpacityChange,
            onDisplayModeChange,
            onPropertyChange,
            onSecondaryTypeChange,
            onSetAllVisibility,
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
        showAllButton,
        hideAllButton,
        exportButton,
        colours,
        visibility,
        opacities,
        onColourChange,
        onVisibilityChange,
        onOpacityChange,
        onDisplayModeChange,
        onPropertyChange,
        onSecondaryTypeChange,
        onSetAllVisibility,
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
            showAllButton,
            hideAllButton,
            exportButton,
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

        expect(showAllButton.disabled).toBe(true);
        expect(hideAllButton.disabled).toBe(true);
        expect(exportButton.disabled).toBe(true);
    });

    it("renders annotation groups and their state", () => {
        // Test rendering annotation groups with their current control state.
        const {
            controller,
            list,
            showAllButton,
            hideAllButton,
            exportButton,
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

        expect(showAllButton.disabled).toBe(false);
        expect(hideAllButton.disabled).toBe(false);
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
        ).toBe(true);
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
            exportButton,
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
            secondaryType: "Tumour",
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
            "Tumour",
            "Stroma",
        ]);

        expect(
            secondaryTypeSelect.value,
        ).toBe(
            JSON.stringify("Tumour"),
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
            "Tumour · prob values · low → high",
        );

        const colours =
            list.querySelectorAll(
                ".annotations-panel-colour",
            );

        expect(
            colours[0].disabled,
        ).toBe(true);

        expect(
            colours[1].disabled,
        ).toBe(false);

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
        ).toBe(false);
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
            "Tumour",
            0.4,
        );
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
            secondaryType: 0,
        });

        controller.render();

        secondaryTypeSelect.value =
            JSON.stringify(1);

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
            1,
        );
    });

    it("calls the bulk visibility callbacks", async () => {
        // Test Show all and Hide all call the bulk visibility callback.
        const {
            controller,
            showAllButton,
            hideAllButton,
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

        showAllButton.click();

        await flushActions();

        expect(
            onSetAllVisibility,
        ).toHaveBeenCalledWith(true);

        hideAllButton.click();

        await flushActions();

        expect(
            onSetAllVisibility,
        ).toHaveBeenCalledWith(false);

        expect(
            onSetAllVisibility,
        ).toHaveBeenCalledTimes(2);
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
