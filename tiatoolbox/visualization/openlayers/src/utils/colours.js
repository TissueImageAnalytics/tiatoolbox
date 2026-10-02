// Shared colour helpers for viewer controls, grids, and themes.
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

export {
    getContrastingColour,
    hexToRgb,
    mixColour,
    toRgba,
};
