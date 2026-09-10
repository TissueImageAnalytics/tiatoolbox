import {
    describe,
    expect,
    it,
} from "vitest";

import {
    assignAnnotationColours,
} from "../../../tiatoolbox/visualization/openlayers/src/utils/annotation-colours.js";

describe("assignAnnotationColours", () => {
    it("assigns different colours to new annotation types", () => {
        const colourMap = new Map();

        assignAnnotationColours(
            colourMap,
            [0, 1, 2],
        );

        expect(colourMap.size).toBe(3);

        for (const colour of colourMap.values()) {
            expect(colour).toHaveLength(4);
            expect(colour[3]).toBe(1);
        }

        expect(
            new Set(
                [...colourMap.values()].map(
                    (colour) =>
                        JSON.stringify(colour),
                ),
            ).size,
        ).toBe(3);
    });

    it("preserves colours already assigned to annotation types", () => {
        const existingColour = [
            0.25,
            0.5,
            0.75,
        ];

        const colourMap = new Map([
            ["tumour", existingColour],
        ]);

        assignAnnotationColours(
            colourMap,
            [
                "tumour",
                "stroma",
            ],
        );

        expect(
            colourMap.get("tumour"),
        ).toBe(existingColour);

        expect(
            colourMap.has("stroma"),
        ).toBe(true);
    });

    it("does not add the same annotation type twice", () => {
        const colourMap = new Map();

        assignAnnotationColours(
            colourMap,
            [
                "tumour",
                "tumour",
            ],
        );

        expect(colourMap.size).toBe(1);
    });

    it("preserves numeric annotation types", () => {
        const colourMap = new Map();

        assignAnnotationColours(
            colourMap,
            [0, 1],
        );

        expect(colourMap.has(0)).toBe(true);
        expect(colourMap.has("0")).toBe(false);
    });

    it("assigns the same colour to each type regardless of load order", () => {
        const firstColourMap =
            new Map();

        const secondColourMap =
            new Map();

        assignAnnotationColours(
            firstColourMap,
            [0, 1, 2],
        );

        assignAnnotationColours(
            firstColourMap,
            [2, 3, 4],
        );

        assignAnnotationColours(
            secondColourMap,
            [2, 3, 4],
        );

        assignAnnotationColours(
            secondColourMap,
            [0, 1, 2],
        );

        for (const annotationType of [
            0,
            1,
            2,
            3,
            4,
        ]) {
            expect(
                firstColourMap.get(
                    annotationType,
                ),
            ).toEqual(
                secondColourMap.get(
                    annotationType,
                ),
            );
        }
    });

    it("assigns stable colours to named annotation types", () => {
        const firstColourMap =
            new Map();

        const secondColourMap =
            new Map();

        assignAnnotationColours(
            firstColourMap,
            [
                "tumour",
                "stroma",
            ],
        );

        assignAnnotationColours(
            secondColourMap,
            [
                "stroma",
                "tumour",
            ],
        );

        expect(
            firstColourMap.get("tumour"),
        ).toEqual(
            secondColourMap.get("tumour"),
        );

        expect(
            firstColourMap.get("stroma"),
        ).toEqual(
            secondColourMap.get("stroma"),
        );
    });

    it("keeps existing colours when new types are discovered", () => {
        const colourMap = new Map();

        assignAnnotationColours(
            colourMap,
            [
                "stroma",
                "tumour",
            ],
        );

        const tumourColour =
            colourMap.get("tumour");

        assignAnnotationColours(
            colourMap,
            [
                "necrosis",
                "tumour",
            ],
        );

        expect(
            colourMap.get("tumour"),
        ).toEqual(tumourColour);

        expect(
            colourMap.has("necrosis"),
        ).toBe(true);
    });
});
