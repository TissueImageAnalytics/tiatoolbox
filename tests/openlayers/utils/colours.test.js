import {
    describe,
    expect,
    it,
} from "vitest";

import {
    getContrastingColour,
    hexToRgb,
    mixColour,
    toRgba,
} from "../../../tiatoolbox/visualization/openlayers/src/utils/colours.js";

describe("hexToRgb", () => {
    it.each([
        [
            "#000000",
            { r: 0, g: 0, b: 0 },
        ],
        [
            "#ffffff",
            { r: 255, g: 255, b: 255 },
        ],
        [
            "#12AbEF",
            { r: 18, g: 171, b: 239 },
        ],
        [
            "336699",
            { r: 51, g: 102, b: 153 },
        ],
    ])(
        "converts %s to RGB values",
        (hex, expected) => {
            expect(
                hexToRgb(hex),
            ).toEqual(expected);
        },
    );

    it.each([
        "#fff",
        "#12345",
        "#1234567",
        "#gggggg",
        "",
    ])(
        "returns null for invalid colour %s",
        (hex) => {
            expect(
                hexToRgb(hex),
            ).toBeNull();
        },
    );
});

describe("getContrastingColour", () => {
    it("returns white for a dark colour", () => {
        expect(
            getContrastingColour({
                r: 0,
                g: 0,
                b: 0,
            }),
        ).toBe("#ffffff");
    });

    it("returns black for a light colour", () => {
        expect(
            getContrastingColour({
                r: 255,
                g: 255,
                b: 255,
            }),
        ).toBe("#000000");
    });
});

describe("mixColour", () => {
    it("returns the original colour when the amount is zero", () => {
        expect(
            mixColour(
                {
                    r: 10,
                    g: 20,
                    b: 30,
                },
                255,
                0,
            ),
        ).toEqual({
            r: 10,
            g: 20,
            b: 30,
        });
    });

    it("returns the target colour when the amount is one", () => {
        expect(
            mixColour(
                {
                    r: 10,
                    g: 20,
                    b: 30,
                },
                255,
                1,
            ),
        ).toEqual({
            r: 255,
            g: 255,
            b: 255,
        });
    });

    it("mixes and rounds colour channels", () => {
        expect(
            mixColour(
                {
                    r: 10,
                    g: 20,
                    b: 30,
                },
                255,
                0.5,
            ),
        ).toEqual({
            r: 133,
            g: 138,
            b: 143,
        });
    });

    it("can mix a colour towards black", () => {
        expect(
            mixColour(
                {
                    r: 100,
                    g: 150,
                    b: 200,
                },
                0,
                0.25,
            ),
        ).toEqual({
            r: 75,
            g: 113,
            b: 150,
        });
    });
});

describe("toRgba", () => {
    it.each([
        [
            {
                r: 10,
                g: 20,
                b: 30,
            },
            0.5,
            "rgba(10, 20, 30, 0.5)",
        ],
        [
            {
                r: 255,
                g: 255,
                b: 255,
            },
            1,
            "rgba(255, 255, 255, 1)",
        ],
        [
            {
                r: 0,
                g: 0,
                b: 0,
            },
            0,
            "rgba(0, 0, 0, 0)",
        ],
    ])(
        "formats RGB values and opacity as rgba",
        (rgb, opacity, expected) => {
            expect(
                toRgba(rgb, opacity),
            ).toBe(expected);
        },
    );
});
