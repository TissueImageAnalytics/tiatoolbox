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
} = {}) {
    document.body.innerHTML = `
        <button id="toggle" type="button"></button>

        <aside id="panel" class="hidden">
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
            showAllButton,
            hideAllButton,
            exportButton,

            getAnnotationGroups: () =>
                annotationGroups,

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
            onSetAllVisibility,
            onExport,
            onOpen,
        });

    return {
        controller,
        panel,
        toggle,
        list,
        showAllButton,
        hideAllButton,
        exportButton,
        colours,
        visibility,
        opacities,
        onColourChange,
        onVisibilityChange,
        onOpacityChange,
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
            showAllButton,
            hideAllButton,
            exportButton,
        } = createHarness();

        controller.render();

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
