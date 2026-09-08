import {
    beforeEach,
    describe,
    expect,
    it,
    vi,
} from "vitest";

import {
    createMapControlsController,
} from "../../../tiatoolbox/visualization/openlayers/src/controls/map-controls.js";

function createCheckbox(
    checked = true,
) {
    const input =
        document.createElement(
            "input",
        );

    input.type = "checkbox";
    input.checked = checked;

    return input;
}

function createSelect(
    values,
    value,
) {
    const select =
        document.createElement(
            "select",
        );

    for (const optionValue of values) {
        const option =
            document.createElement(
                "option",
            );

        option.value = optionValue;
        option.textContent =
            optionValue;

        select.append(option);
    }

    select.value = value;

    return select;
}

function createSource({
    extent = [
        0,
        0,
        1000,
        800,
    ],
} = {}) {
    return {
        getTileGrid: vi.fn(
            () => ({
                getExtent: vi.fn(
                    () => extent,
                ),
            }),
        ),
    };
}

function dispatchChange(element) {
    element.dispatchEvent(
        new Event(
            "change",
            {
                bubbles: true,
            },
        ),
    );
}

function dispatchBlur(element) {
    element.dispatchEvent(
        new Event(
            "blur",
        ),
    );
}

function dispatchFocus(element) {
    element.dispatchEvent(
        new Event(
            "focus",
        ),
    );
}

function dispatchKeydown(
    element,
    key,
) {
    const event =
        new KeyboardEvent(
            "keydown",
            {
                key,
                bubbles: true,
                cancelable: true,
            },
        );

    element.dispatchEvent(
        event,
    );

    return event;
}

function createHarness({
    slideLoaded = true,
    zoom = 4,
    zoomVisible = true,
    zoomLevelVisible = true,
    rotationVisible = true,
    resetViewVisible = true,
    fullscreenVisible = true,
    mousePositionVisible = true,
    mouseWheelSensitivity = "default",
    zoomButtonStep = "1",
} = {}) {
    const state = {
        source:
            slideLoaded
                ? createSource()
                : null,
        zoom,
    };

    const view = {
        getZoom: vi.fn(
            () => state.zoom,
        ),

        getMinZoom: vi.fn(
            () => 0,
        ),

        getMaxZoom: vi.fn(
            () => 10,
        ),

        setZoom: vi.fn(
            (newZoom) => {
                state.zoom = newZoom;
            },
        ),

        setRotation: vi.fn(),

        fit: vi.fn(),
    };

    const map = {
        addInteraction:
            vi.fn(),

        removeInteraction:
            vi.fn(),

        addControl:
            vi.fn(),

        removeControl: vi.fn(
            (control) => {
                control.element.remove();
            },
        ),

        getView: vi.fn(
            () => view,
        ),

        getSize: vi.fn(
            () => [
                800,
                600,
            ],
        ),
    };

    const viewerApp =
        document.createElement(
            "div",
        );

    const resetViewControl =
        document.createElement(
            "div",
        );

    resetViewControl.className =
        "reset-view-control";

    const resetViewButton =
        document.createElement(
            "button",
        );

    resetViewButton.type = "button";

    resetViewControl.append(
        resetViewButton,
    );

    const zoomVisibleInput =
        createCheckbox(
            zoomVisible,
        );

    const zoomLevelVisibleInput =
        createCheckbox(
            zoomLevelVisible,
        );

    const rotationVisibleInput =
        createCheckbox(
            rotationVisible,
        );

    const resetViewVisibleInput =
        createCheckbox(
            resetViewVisible,
        );

    const fullscreenVisibleInput =
        createCheckbox(
            fullscreenVisible,
        );

    const mousePositionVisibleInput =
        createCheckbox(
            mousePositionVisible,
        );

    const mouseWheelZoomSensitivitySelect =
        createSelect(
            [
                "low",
                "default",
                "high",
            ],
            mouseWheelSensitivity,
        );

    const zoomButtonStepSelect =
        createSelect(
            [
                "0.1",
                "0.5",
                "1",
                "2",
            ],
            zoomButtonStep,
        );

    document.body.append(
        viewerApp,
        zoomVisibleInput,
        zoomLevelVisibleInput,
        rotationVisibleInput,
        resetViewVisibleInput,
        fullscreenVisibleInput,
        mousePositionVisibleInput,
        mouseWheelZoomSensitivitySelect,
        zoomButtonStepSelect,
    );

    const controller =
        createMapControlsController({
            map,
            viewerApp,

            getSlideSource: () =>
                state.source,

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

    const zoomControl =
        viewerApp.querySelector(
            ".ol-zoom",
        );

    const zoomLevel =
        viewerApp.querySelector(
            ".ol-zoom-level",
        );

    const bottomControlsGroup =
        viewerApp.querySelector(
            ".bottom-controls-group",
        );

    const mouseWheelInteraction =
        map.addInteraction
            .mock.calls[0][0];

    const zoomControlInstance =
        map.addControl
            .mock.calls[0][0];

    return {
        state,
        map,
        view,
        viewerApp,
        resetViewButton,
        resetViewControl,
        zoomVisibleInput,
        zoomLevelVisibleInput,
        rotationVisibleInput,
        resetViewVisibleInput,
        fullscreenVisibleInput,
        mousePositionVisibleInput,
        mouseWheelZoomSensitivitySelect,
        zoomButtonStepSelect,
        controller,
        zoomControl,
        zoomLevel,
        bottomControlsGroup,
        mouseWheelInteraction,
        zoomControlInstance,
    };
}

beforeEach(() => {
    document.body.replaceChildren();
});

describe("initialisation", () => {
    it("creates the map controls and mouse wheel interaction", () => {
        const {
            map,
            controller,
            zoomControlInstance,
            mouseWheelInteraction,
        } = createHarness();

        expect(
            map.addInteraction,
        ).toHaveBeenCalledOnce();

        expect(
            map.addInteraction,
        ).toHaveBeenCalledWith(
            mouseWheelInteraction,
        );

        expect(
            map.addControl,
        ).toHaveBeenCalledTimes(
            4,
        );

        expect(
            map.addControl,
        ).toHaveBeenNthCalledWith(
            1,
            zoomControlInstance,
        );

        expect(
            map.addControl,
        ).toHaveBeenNthCalledWith(
            2,
            controller.mousePositionControl,
        );

        expect(
            map.addControl,
        ).toHaveBeenNthCalledWith(
            3,
            controller.rotate,
        );

        expect(
            map.addControl,
        ).toHaveBeenNthCalledWith(
            4,
            controller.fullscreen,
        );
    });

    it("builds the bottom controls group in the expected order", () => {
        const {
            bottomControlsGroup,
            resetViewControl,
            zoomControl,
            controller,
        } = createHarness();

        expect(
            bottomControlsGroup,
        ).not.toBeNull();

        expect([
            ...bottomControlsGroup
                .children,
        ]).toEqual([
            resetViewControl,
            zoomControl,
            controller.fullscreen
                .element,
        ]);
    });

    it("inserts the zoom level input before the zoom-out button", () => {
        const {
            zoomControl,
            zoomLevel,
        } = createHarness();

        expect(
            zoomLevel,
        ).not.toBeNull();

        expect(
            zoomLevel.type,
        ).toBe("number");

        expect(
            zoomLevel.className,
        ).toBe("ol-zoom-level");

        expect(
            zoomLevel.step,
        ).toBe("1");

        expect(
            zoomLevel.getAttribute(
                "aria-label",
            ),
        ).toBe("Zoom level");

        expect(
            zoomLevel.title,
        ).toBe("Zoom level");

        expect(
            zoomLevel.nextElementSibling,
        ).toBe(
            zoomControl.querySelector(
                ".ol-zoom-out",
            ),
        );
    });

    it("displays an integer zoom level without decimals", () => {
        const {
            zoomLevel,
        } = createHarness({
            zoom: 4,
        });

        expect(
            zoomLevel.value,
        ).toBe("4");
    });

    it("displays a fractional zoom level with one decimal place", () => {
        const {
            zoomLevel,
        } = createHarness({
            zoom: 4.25,
        });

        expect(
            zoomLevel.value,
        ).toBe("4.3");
    });

    it("clears the zoom level when the view has no zoom", () => {
        const {
            state,
            controller,
            zoomLevel,
        } = createHarness();

        state.zoom = undefined;

        controller.updateZoomLevel();

        expect(
            zoomLevel.value,
        ).toBe("");
    });

    it("formats mouse coordinates with an inverted y-axis", () => {
        const {
            controller,
        } = createHarness();

        const coordinateFormat =
            controller
                .mousePositionControl
                .getCoordinateFormat();

        expect(
            coordinateFormat([
                12,
                -34,
            ]),
        ).toBe("12, 34");
    });
});

describe("zoom level input", () => {
    it("selects the zoom level when focused", () => {
        const {
            zoomLevel,
        } = createHarness();

        const select =
            vi.spyOn(
                zoomLevel,
                "select",
            );

        dispatchFocus(
            zoomLevel,
        );

        expect(
            select,
        ).toHaveBeenCalledOnce();
    });

    it("applies a valid zoom level on blur", () => {
        const {
            zoomLevel,
            view,
        } = createHarness();

        zoomLevel.value = "6.5";

        dispatchBlur(
            zoomLevel,
        );

        expect(
            view.setZoom,
        ).toHaveBeenCalledOnce();

        expect(
            view.setZoom,
        ).toHaveBeenCalledWith(
            6.5,
        );

        expect(
            zoomLevel.value,
        ).toBe("6.5");
    });

    it.each([
        [
            "-5",
            0,
        ],
        [
            "20",
            10,
        ],
    ])(
        "clamps zoom level %s to %s",
        (
            enteredZoom,
            expectedZoom,
        ) => {
            const {
                zoomLevel,
                view,
            } = createHarness();

            zoomLevel.value =
                enteredZoom;

            dispatchBlur(
                zoomLevel,
            );

            expect(
                view.setZoom,
            ).toHaveBeenCalledWith(
                expectedZoom,
            );

            expect(
                zoomLevel.value,
            ).toBe(
                expectedZoom.toString(),
            );
        },
    );

    it("restores the current zoom when the input is invalid", () => {
        const {
            zoomLevel,
            view,
        } = createHarness({
            zoom: 4.5,
        });

        zoomLevel.value = "";

        dispatchBlur(
            zoomLevel,
        );

        expect(
            view.setZoom,
        ).not.toHaveBeenCalled();

        expect(
            zoomLevel.value,
        ).toBe("4.5");
    });

    it("applies the zoom level when Enter is pressed", () => {
        const {
            zoomLevel,
            view,
        } = createHarness();

        zoomLevel.focus();
        zoomLevel.value = "7";

        const event =
            dispatchKeydown(
                zoomLevel,
                "Enter",
            );

        expect(
            event.defaultPrevented,
        ).toBe(true);

        expect(
            view.setZoom,
        ).toHaveBeenCalledWith(
            7,
        );

        expect(
            zoomLevel.value,
        ).toBe("7");
    });

    it("restores the current zoom when Escape is pressed", () => {
        const {
            zoomLevel,
        } = createHarness({
            zoom: 4.5,
        });

        zoomLevel.focus();
        zoomLevel.value = "8";

        const event =
            dispatchKeydown(
                zoomLevel,
                "Escape",
            );

        expect(
            event.defaultPrevented,
        ).toBe(true);

        expect(
            zoomLevel.value,
        ).toBe("4.5");
    });
});

describe("reset view", () => {
    it("does nothing when no slide is loaded", () => {
        const {
            resetViewButton,
            view,
        } = createHarness({
            slideLoaded: false,
        });

        resetViewButton.click();

        expect(
            view.setRotation,
        ).not.toHaveBeenCalled();

        expect(
            view.fit,
        ).not.toHaveBeenCalled();
    });

    it("resets rotation and fits the loaded slide", () => {
        const {
            map,
            resetViewButton,
            view,
        } = createHarness();

        resetViewButton.click();

        expect(
            view.setRotation,
        ).toHaveBeenCalledOnce();

        expect(
            view.setRotation,
        ).toHaveBeenCalledWith(
            0,
        );

        expect(
            view.fit,
        ).toHaveBeenCalledOnce();

        expect(
            view.fit,
        ).toHaveBeenCalledWith(
            [
                0,
                0,
                1000,
                800,
            ],
            {
                size: [
                    800,
                    600,
                ],
            },
        );

        expect(
            map.getSize,
        ).toHaveBeenCalledOnce();
    });
});

describe("viewer state and visibility", () => {
    it("disables interactive controls when the viewer is disabled", () => {
        const {
            resetViewButton,
            zoomControl,
            zoomLevel,
            mouseWheelInteraction,
            controller,
        } = createHarness();

        controller.setViewerEnabled(
            false,
        );

        expect(
            zoomControl.querySelector(
                ".ol-zoom-in",
            ).disabled,
        ).toBe(true);

        expect(
            zoomControl.querySelector(
                ".ol-zoom-out",
            ).disabled,
        ).toBe(true);

        expect(
            controller.rotate.element
                .querySelector(
                    "button",
                ).disabled,
        ).toBe(true);

        expect(
            resetViewButton.disabled,
        ).toBe(true);

        expect(
            zoomLevel.disabled,
        ).toBe(true);

        expect(
            zoomControl.classList
                .contains(
                    "viewer-control-disabled",
                ),
        ).toBe(true);

        expect(
            mouseWheelInteraction
                .getActive(),
        ).toBe(false);

        expect(
            controller
                .mousePositionControl
                .element.classList
                .contains(
                    "viewer-control-hidden",
                ),
        ).toBe(true);
    });

    it("reenables interactive controls when the viewer is enabled", () => {
        const {
            resetViewButton,
            zoomControl,
            zoomLevel,
            mouseWheelInteraction,
            controller,
        } = createHarness();

        controller.setViewerEnabled(
            false,
        );

        controller.setViewerEnabled(
            true,
        );

        expect(
            zoomControl.querySelector(
                ".ol-zoom-in",
            ).disabled,
        ).toBe(false);

        expect(
            zoomControl.querySelector(
                ".ol-zoom-out",
            ).disabled,
        ).toBe(false);

        expect(
            controller.rotate.element
                .querySelector(
                    "button",
                ).disabled,
        ).toBe(false);

        expect(
            resetViewButton.disabled,
        ).toBe(false);

        expect(
            zoomLevel.disabled,
        ).toBe(false);

        expect(
            zoomControl.classList
                .contains(
                    "viewer-control-disabled",
                ),
        ).toBe(false);

        expect(
            mouseWheelInteraction
                .getActive(),
        ).toBe(true);

        expect(
            controller
                .mousePositionControl
                .element.classList
                .contains(
                    "viewer-control-hidden",
                ),
        ).toBe(false);
    });

    it("hides controls when their visibility settings are disabled", () => {
        const harness =
            createHarness();

        harness.zoomVisibleInput.checked =
            false;

        harness.zoomLevelVisibleInput.checked =
            false;

        harness.rotationVisibleInput.checked =
            false;

        harness.resetViewVisibleInput.checked =
            false;

        harness.fullscreenVisibleInput.checked =
            false;

        harness.mousePositionVisibleInput.checked =
            false;

        harness.controller.updateVisibility();

        expect(
            harness.zoomControl
                .classList.contains(
                    "viewer-control-hidden",
                ),
        ).toBe(true);

        expect(
            harness.zoomLevel
                .classList.contains(
                    "viewer-control-hidden",
                ),
        ).toBe(true);

        expect(
            harness.controller
                .rotate.element.classList
                .contains(
                    "viewer-control-hidden",
                ),
        ).toBe(true);

        expect(
            harness.resetViewControl
                .classList.contains(
                    "viewer-control-hidden",
                ),
        ).toBe(true);

        expect(
            harness.controller
                .fullscreen.element
                .classList.contains(
                    "viewer-control-hidden",
                ),
        ).toBe(true);

        expect(
            harness.controller
                .mousePositionControl
                .element.classList
                .contains(
                    "viewer-control-hidden",
                ),
        ).toBe(true);
    });

    it("shows controls when their visibility settings are enabled", () => {
        const harness =
            createHarness({
                zoomVisible: false,
                zoomLevelVisible: false,
                rotationVisible: false,
                resetViewVisible: false,
                fullscreenVisible: false,
                mousePositionVisible: false,
            });

        harness.controller.updateVisibility();

        harness.zoomVisibleInput.checked =
            true;

        harness.zoomLevelVisibleInput.checked =
            true;

        harness.rotationVisibleInput.checked =
            true;

        harness.resetViewVisibleInput.checked =
            true;

        harness.fullscreenVisibleInput.checked =
            true;

        harness.mousePositionVisibleInput.checked =
            true;

        harness.controller.updateVisibility();

        expect(
            harness.zoomControl
                .classList.contains(
                    "viewer-control-hidden",
                ),
        ).toBe(false);

        expect(
            harness.zoomLevel
                .classList.contains(
                    "viewer-control-hidden",
                ),
        ).toBe(false);

        expect(
            harness.controller
                .rotate.element.classList
                .contains(
                    "viewer-control-hidden",
                ),
        ).toBe(false);

        expect(
            harness.resetViewControl
                .classList.contains(
                    "viewer-control-hidden",
                ),
        ).toBe(false);

        expect(
            harness.controller
                .fullscreen.element
                .classList.contains(
                    "viewer-control-hidden",
                ),
        ).toBe(false);

        expect(
            harness.controller
                .mousePositionControl
                .element.classList
                .contains(
                    "viewer-control-hidden",
                ),
        ).toBe(false);
    });

    it("keeps mouse position hidden when no slide is loaded", () => {
        const {
            controller,
        } = createHarness({
            slideLoaded: false,
            mousePositionVisible: true,
        });

        controller.updateVisibility();

        expect(
            controller
                .mousePositionControl
                .element.classList
                .contains(
                    "viewer-control-hidden",
                ),
        ).toBe(true);
    });

    it("keeps mouse position hidden when its setting is disabled", () => {
        const {
            controller,
        } = createHarness({
            mousePositionVisible: false,
        });

        controller.setViewerEnabled(
            true,
        );

        expect(
            controller
                .mousePositionControl
                .element.classList
                .contains(
                    "viewer-control-hidden",
                ),
        ).toBe(true);
    });
});

describe("mouse wheel zoom sensitivity", () => {
    it.each([
        [
            "low",
            600,
            1,
        ],
        [
            "default",
            300,
            1,
        ],
        [
            "high",
            150,
            2,
        ],
    ])(
        "uses the %s mouse wheel zoom preset",
        (
            preset,
            expectedDeltaPerZoom,
            expectedMaxDelta,
        ) => {
            const {
                mouseWheelInteraction,
            } = createHarness({
                mouseWheelSensitivity:
                    preset,
            });

            expect(
                mouseWheelInteraction
                    .deltaPerZoom_,
            ).toBe(
                expectedDeltaPerZoom,
            );

            expect(
                mouseWheelInteraction
                    .maxDelta_,
            ).toBe(
                expectedMaxDelta,
            );
        },
    );

    it("falls back to the default mouse wheel zoom preset", () => {
        const {
            mouseWheelInteraction,
        } = createHarness({
            mouseWheelSensitivity: "",
        });

        expect(
            mouseWheelInteraction
                .deltaPerZoom_,
        ).toBe(300);

        expect(
            mouseWheelInteraction
                .maxDelta_,
        ).toBe(1);
    });

    it("starts with mouse wheel zoom disabled when no slide is loaded", () => {
        const {
            mouseWheelInteraction,
        } = createHarness({
            slideLoaded: false,
        });

        expect(
            mouseWheelInteraction
                .getActive(),
        ).toBe(false);
    });

    it("replaces the mouse wheel interaction when sensitivity changes", () => {
        const {
            map,
            mouseWheelInteraction,
            mouseWheelZoomSensitivitySelect,
        } = createHarness();

        mouseWheelZoomSensitivitySelect.value =
            "high";

        dispatchChange(
            mouseWheelZoomSensitivitySelect,
        );

        const replacementInteraction =
            map.addInteraction
                .mock.calls.at(-1)[0];

        expect(
            replacementInteraction,
        ).not.toBe(
            mouseWheelInteraction,
        );

        expect(
            map.removeInteraction,
        ).toHaveBeenCalledOnce();

        expect(
            map.removeInteraction,
        ).toHaveBeenCalledWith(
            mouseWheelInteraction,
        );

        expect(
            map.addInteraction,
        ).toHaveBeenCalledTimes(
            2,
        );

        expect(
            replacementInteraction
                .deltaPerZoom_,
        ).toBe(150);

        expect(
            replacementInteraction
                .maxDelta_,
        ).toBe(2);

        expect(
            replacementInteraction
                .getActive(),
        ).toBe(true);
    });

    it("keeps a replacement mouse wheel interaction disabled when the viewer is disabled", () => {
        const {
            map,
            controller,
            mouseWheelZoomSensitivitySelect,
        } = createHarness();

        controller.setViewerEnabled(
            false,
        );

        mouseWheelZoomSensitivitySelect.value =
            "high";

        dispatchChange(
            mouseWheelZoomSensitivitySelect,
        );

        const replacementInteraction =
            map.addInteraction
                .mock.calls.at(-1)[0];

        expect(
            replacementInteraction
                .getActive(),
        ).toBe(false);
    });
});

describe("zoom button step", () => {
    it("recreates the zoom control when the step changes", () => {
        const {
            map,
            viewerApp,
            zoomLevel,
            zoomControl,
            zoomControlInstance,
            zoomButtonStepSelect,
            controller,
        } = createHarness();

        zoomButtonStepSelect.value =
            "0.5";

        dispatchChange(
            zoomButtonStepSelect,
        );

        const replacementControl =
            map.addControl
                .mock.calls.at(-1)[0];

        const replacementElement =
            viewerApp.querySelector(
                ".ol-zoom",
            );

        expect(
            replacementControl,
        ).not.toBe(
            zoomControlInstance,
        );

        expect(
            replacementElement,
        ).not.toBe(
            zoomControl,
        );

        expect(
            map.removeControl,
        ).toHaveBeenCalledOnce();

        expect(
            map.removeControl,
        ).toHaveBeenCalledWith(
            zoomControlInstance,
        );

        expect(
            map.addControl,
        ).toHaveBeenCalledTimes(
            5,
        );

        expect(
            replacementElement
                .contains(
                    zoomLevel,
                ),
        ).toBe(true);

        expect(
            zoomLevel.nextElementSibling,
        ).toBe(
            replacementElement
                .querySelector(
                    ".ol-zoom-out",
                ),
        );

        expect(
            replacementElement
                .nextElementSibling,
        ).toBe(
            controller.fullscreen
                .element,
        );
    });

    it("preserves viewer state and visibility when the zoom control is recreated", () => {
        const {
            viewerApp,
            zoomLevel,
            zoomVisibleInput,
            zoomButtonStepSelect,
            controller,
        } = createHarness();

        zoomVisibleInput.checked =
            false;

        controller.updateVisibility();

        controller.setViewerEnabled(
            false,
        );

        zoomButtonStepSelect.value =
            "2";

        dispatchChange(
            zoomButtonStepSelect,
        );

        const replacementElement =
            viewerApp.querySelector(
                ".ol-zoom",
            );

        expect(
            replacementElement
                .classList.contains(
                    "viewer-control-hidden",
                ),
        ).toBe(true);

        expect(
            replacementElement
                .classList.contains(
                    "viewer-control-disabled",
                ),
        ).toBe(true);

        expect(
            replacementElement
                .querySelector(
                    ".ol-zoom-in",
                ).disabled,
        ).toBe(true);

        expect(
            replacementElement
                .querySelector(
                    ".ol-zoom-out",
                ).disabled,
        ).toBe(true);

        expect(
            zoomLevel.disabled,
        ).toBe(true);
    });
});
