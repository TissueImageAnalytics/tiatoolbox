import ScaleLine from "ol/control/ScaleLine.js";

import {
    getContrastingColour,
    hexToRgb,
    toRgba,
} from "../utils/colours.js";

const scaleBarWidths = {
    small: 70,
    default: 100,
    large: 140,
};

// Manage scale bar visibility, appearance, size and units.
function createScaleBarController({
    map,
    hasSlide,
    enabledInput,
    colourInput,
    opacityInput,
    opacityValue,
    sizeSelect,
    unitsSelect,
    onControlChange,
}) {
    function createControl() {
        const minWidth =
            scaleBarWidths[sizeSelect.value] ??
            scaleBarWidths.default;

        return new ScaleLine({
            units: unitsSelect.value,
            minWidth,
        });
    }

    let control = createControl();

    map.addControl(control);
    onControlChange(control);

    function updateVisibility() {
        control.element.classList.toggle(
            "viewer-control-hidden",
            !hasSlide() || !enabledInput.checked,
        );
    }

    function updateColour() {
        const colour = colourInput.value;

        const scaleLineInner = control.element.querySelector(
            ".ol-scale-line-inner",
        );

        if (scaleLineInner === null) {
            return;
        }

        scaleLineInner.style.color = colour;
        scaleLineInner.style.borderColor = colour;
    }

    function updateOpacity() {
        const opacity =
            Number(opacityInput.value) / 100;

        const scaleBarColour =
            hexToRgb(colourInput.value);

        if (scaleBarColour === null) {
            return;
        }

        const contrastColour =
            getContrastingColour(scaleBarColour);

        const backgroundColour =
            contrastColour === "#ffffff"
                ? { r: 255, g: 255, b: 255 }
                : { r: 17, g: 17, b: 17 };

        control.element.style.backgroundColor =
            toRgba(backgroundColour, opacity);

        opacityValue.textContent =
            `${opacityInput.value}%`;
    }

    function updateSize() {
        map.removeControl(control);

        control = createControl();

        map.addControl(control);
        onControlChange(control);

        updateVisibility();
        updateColour();
        updateOpacity();
    }

    function updateUnits() {
        control.setUnits(unitsSelect.value);
    }

    function setViewerEnabled(enabled) {
        control.element.classList.toggle(
            "viewer-control-hidden",
            !enabled || !enabledInput.checked,
        );
    }

    enabledInput.addEventListener("change", () => {
        updateVisibility();
    });

    colourInput.addEventListener("input", () => {
        updateColour();
        updateOpacity();
    });

    opacityInput.addEventListener("input", () => {
        updateOpacity();
    });

    sizeSelect.addEventListener("change", () => {
        updateSize();
    });

    unitsSelect.addEventListener("change", () => {
        updateUnits();
    });

    updateColour();
    updateOpacity();

    return {
        setViewerEnabled,
        updateColour,
        updateOpacity,
        updateSize,
        updateUnits,
        updateVisibility,
    };
}

export { createScaleBarController };
