import {
    beforeEach,
    describe,
    expect,
    it,
    vi,
} from "vitest";

import {
    createScaleBarController,
} from "../../../tiatoolbox/visualization/openlayers/src/controls/scale-bar.js";

let currentControl = null;

function createCheckbox(
    checked = true,
) {
    const input =
        document.createElement("input");

    input.type = "checkbox";
    input.checked = checked;

    return input;
}

function createInput(
    value,
    type = "range",
) {
    const input =
        document.createElement("input");

    input.type = type;

    if (type === "range") {
        input.min = "0";
        input.max = "100";
    }

    input.value = value;

    return input;
}

function createSelect(
    values,
    value,
) {
    const select =
        document.createElement("select");

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

function dispatchInput(element) {
    element.dispatchEvent(
        new Event(
            "input",
            {
                bubbles: true,
            },
        ),
    );
}

function createHarness({
    slideLoaded = true,
    enabled = true,
    colour = "#ffffff",
    opacity = "100",
    size = "default",
    units = "metric",
} = {}) {
    const state = {
        slideLoaded,
    };

    const map = {
        addControl: vi.fn(),
        removeControl: vi.fn(),
    };

    const enabledInput =
        createCheckbox(enabled);

    const colourInput =
        createInput(
            colour,
            "color",
        );

    const opacityInput =
        createInput(opacity);

    const opacityValue =
        document.createElement(
            "span",
        );

    const sizeSelect =
        createSelect(
            [
                "small",
                "default",
                "large",
            ],
            size,
        );

    const unitsSelect =
        createSelect(
            [
                "metric",
                "imperial",
            ],
            units,
        );

    document.body.append(
        enabledInput,
        colourInput,
        opacityInput,
        opacityValue,
        sizeSelect,
        unitsSelect,
    );

    const onControlChange =
        vi.fn((control) => {
            currentControl = control;
        });

    const controller =
        createScaleBarController({
            map,
            hasSlide: () =>
                state.slideLoaded,
            enabledInput,
            colourInput,
            opacityInput,
            opacityValue,
            sizeSelect,
            unitsSelect,
            onControlChange,
        });

    return {
        state,
        map,
        enabledInput,
        colourInput,
        opacityInput,
        opacityValue,
        sizeSelect,
        unitsSelect,
        onControlChange,
        controller,
    };
}

function getCurrentControl() {
    return currentControl;
}

function getScaleLineInner(
    control,
) {
    return control.element.querySelector(
        ".ol-scale-line-inner",
    );
}

beforeEach(() => {
    document.body.replaceChildren();

    currentControl = null;
});

describe("initialisation", () => {
    it("creates and adds the scale bar control", () => {
        const {
            map,
            onControlChange,
        } = createHarness();

        const control =
            getCurrentControl();

        expect(control).not.toBeNull();

        expect(
            control.getUnits(),
        ).toBe("metric");

        expect(
            control.minWidth_,
        ).toBe(100);

        expect(
            map.addControl,
        ).toHaveBeenCalledExactlyOnceWith(
            control,
        );

        expect(
            onControlChange,
        ).toHaveBeenCalledExactlyOnceWith(
            control,
        );
    });

    it.each([
        [
            "small",
            70,
        ],
        [
            "default",
            100,
        ],
        [
            "large",
            140,
        ],
    ])(
        "uses the %s scale bar width",
        (size, expectedWidth) => {
            createHarness({
                size,
            });

            expect(
                getCurrentControl()
                    .minWidth_,
            ).toBe(expectedWidth);
        },
    );

    it("falls back to the default width for an unknown size", () => {
        const harness =
            createHarness();

        harness.sizeSelect.value = "";

        harness.controller.updateSize();

        expect(
            getCurrentControl()
                .minWidth_,
        ).toBe(100);
    });

    it("applies the initial colour and opacity", () => {
        const {
            opacityValue,
        } = createHarness({
            colour: "#123456",
            opacity: "65",
        });

        const control =
            getCurrentControl();

        const inner =
            getScaleLineInner(
                control,
            );

        expect(
            inner.style.color,
        ).toBe(
            "rgb(18, 52, 86)",
        );

        expect(
            inner.style.borderColor,
        ).toBe(
            "rgb(18, 52, 86)",
        );

        expect(
            control.element.style
                .backgroundColor,
        ).toBe(
            "rgba(255, 255, 255, 0.65)",
        );

        expect(
            opacityValue.textContent,
        ).toBe("65%");
    });
});

describe("visibility", () => {
    it("shows the control when a slide is loaded and the scale bar is enabled", () => {
        const {
            controller,
        } = createHarness();

        controller.updateVisibility();

        expect(
            getCurrentControl()
                .element.classList.contains(
                    "viewer-control-hidden",
                ),
        ).toBe(false);
    });

    it("hides the control when no slide is loaded", () => {
        const {
            controller,
        } = createHarness({
            slideLoaded: false,
        });

        controller.updateVisibility();

        expect(
            getCurrentControl()
                .element.classList.contains(
                    "viewer-control-hidden",
                ),
        ).toBe(true);
    });

    it("hides the control when the scale bar is disabled", () => {
        const {
            controller,
        } = createHarness({
            enabled: false,
        });

        controller.updateVisibility();

        expect(
            getCurrentControl()
                .element.classList.contains(
                    "viewer-control-hidden",
                ),
        ).toBe(true);
    });

    it("updates visibility when the enabled input changes", () => {
        const {
            enabledInput,
        } = createHarness();

        enabledInput.checked = false;

        dispatchChange(
            enabledInput,
        );

        expect(
            getCurrentControl()
                .element.classList.contains(
                    "viewer-control-hidden",
                ),
        ).toBe(true);
    });

    it("respects viewer enabled state", () => {
        const {
            controller,
        } = createHarness();

        controller.setViewerEnabled(
            false,
        );

        expect(
            getCurrentControl()
                .element.classList.contains(
                    "viewer-control-hidden",
                ),
        ).toBe(true);

        controller.setViewerEnabled(
            true,
        );

        expect(
            getCurrentControl()
                .element.classList.contains(
                    "viewer-control-hidden",
                ),
        ).toBe(false);
    });

    it("stays hidden when the viewer is enabled but the scale bar setting is disabled", () => {
        const {
            controller,
        } = createHarness({
            enabled: false,
        });

        controller.setViewerEnabled(
            true,
        );

        expect(
            getCurrentControl()
                .element.classList.contains(
                    "viewer-control-hidden",
                ),
        ).toBe(true);
    });
});

describe("appearance", () => {
    it("updates the scale bar colour", () => {
        const {
            colourInput,
        } = createHarness();

        colourInput.value =
            "#ff0000";

        dispatchInput(
            colourInput,
        );

        const inner =
            getScaleLineInner(
                getCurrentControl(),
            );

        expect(
            inner.style.color,
        ).toBe(
            "rgb(255, 0, 0)",
        );

        expect(
            inner.style.borderColor,
        ).toBe(
            "rgb(255, 0, 0)",
        );
    });

    it("uses a light background for a dark scale bar colour", () => {
        const {
            colourInput,
            opacityInput,
            opacityValue,
        } = createHarness();

        colourInput.value =
            "#000000";

        opacityInput.value = "50";

        dispatchInput(
            opacityInput,
        );

        expect(
            getCurrentControl()
                .element.style
                .backgroundColor,
        ).toBe(
            "rgba(255, 255, 255, 0.5)",
        );

        expect(
            opacityValue.textContent,
        ).toBe("50%");
    });

    it("uses a dark background for a light scale bar colour", () => {
        const {
            opacityInput,
        } = createHarness({
            colour: "#ffffff",
        });

        opacityInput.value = "75";

        dispatchInput(
            opacityInput,
        );

        expect(
            getCurrentControl()
                .element.style
                .backgroundColor,
        ).toBe(
            "rgba(17, 17, 17, 0.75)",
        );
    });

    it("returns without changing opacity styling for an invalid colour", () => {
        const {
            colourInput,
            opacityInput,
            opacityValue,
            controller,
        } = createHarness();

        const control =
            getCurrentControl();

        const previousBackground =
            control.element.style
                .backgroundColor;

        Object.defineProperty(
            colourInput,
            "value",
            {
                configurable: true,
                value: "invalid",
            },
        );

        opacityInput.value = "40";

        controller.updateOpacity();

        expect(
            control.element.style
                .backgroundColor,
        ).toBe(
            previousBackground,
        );

        expect(
            opacityValue.textContent,
        ).toBe("100%");
    });

    it("returns safely when the scale line inner element is missing", () => {
        const {
            colourInput,
            controller,
        } = createHarness();

        const control =
            getCurrentControl();

        control.element.replaceChildren();

        colourInput.value =
            "#ff0000";

        expect(() => {
            controller.updateColour();
        }).not.toThrow();
    });
});

describe("size and units", () => {
    it("recreates the control when size changes", () => {
        const {
            map,
            sizeSelect,
            onControlChange,
        } = createHarness();

        const originalControl =
            getCurrentControl();

        sizeSelect.value =
            "large";

        dispatchChange(
            sizeSelect,
        );

        const replacementControl =
            getCurrentControl();

        expect(
            replacementControl,
        ).not.toBe(
            originalControl,
        );

        expect(
            replacementControl
                .minWidth_,
        ).toBe(140);

        expect(
            map.removeControl,
        ).toHaveBeenCalledExactlyOnceWith(
            originalControl,
        );

        expect(
            map.addControl,
        ).toHaveBeenCalledTimes(
            2,
        );

        expect(
            map.addControl,
        ).toHaveBeenLastCalledWith(
            replacementControl,
        );

        expect(
            onControlChange,
        ).toHaveBeenCalledTimes(
            2,
        );

        expect(
            onControlChange,
        ).toHaveBeenLastCalledWith(
            replacementControl,
        );
    });

    it("reapplies state and appearance after recreating the control", () => {
        const {
            colourInput,
            opacityInput,
            enabledInput,
            sizeSelect,
        } = createHarness();

        colourInput.value =
            "#ff0000";

        opacityInput.value = "40";

        enabledInput.checked =
            false;

        sizeSelect.value =
            "small";

        dispatchChange(
            sizeSelect,
        );

        const control =
            getCurrentControl();

        const inner =
            getScaleLineInner(
                control,
            );

        expect(
            control.minWidth_,
        ).toBe(70);

        expect(
            control.element.classList
                .contains(
                    "viewer-control-hidden",
                ),
        ).toBe(true);

        expect(
            inner.style.color,
        ).toBe(
            "rgb(255, 0, 0)",
        );

        expect(
            control.element.style
                .backgroundColor,
        ).toBe(
            "rgba(17, 17, 17, 0.4)",
        );
    });

    it("updates units without recreating the control", () => {
        const {
            unitsSelect,
            map,
        } = createHarness();

        const control =
            getCurrentControl();

        unitsSelect.value =
            "imperial";

        dispatchChange(
            unitsSelect,
        );

        expect(
            control.getUnits(),
        ).toBe("imperial");

        expect(
            map.removeControl,
        ).not.toHaveBeenCalled();

        expect(
            getCurrentControl(),
        ).toBe(control);
    });
});
