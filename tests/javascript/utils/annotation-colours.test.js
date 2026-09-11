import {
    describe,
    expect,
    it,
    vi,
} from "vitest";

import {
    assignAnnotationColours,
} from "../../../tiatoolbox/visualization/openlayers/src/utils/annotation-colours.js";

function createColourGenerator() {
    const colours = new Map([
        [
            0,
            [1, 0, 0, 1],
        ],
        [
            1,
            [0, 1, 0, 1],
        ],
        [
            2,
            [0, 0, 1, 1],
        ],
        [
            "tumour",
            [1, 0.5, 0, 1],
        ],
        [
            "stroma",
            [0, 1, 1, 1],
        ],
    ]);

    return vi.fn(
        async (types) =>
            new Map(
                types.map(
                    (type) => [
                        type,
                        colours.get(type),
                    ],
                ),
            ),
    );
}

describe("assignAnnotationColours", () => {
    it("adds generated colours for new annotation types", async () => {
        const colourMap = new Map();
        const getColours =
            createColourGenerator();

        await assignAnnotationColours(
            colourMap,
            [0, 1, 2],
            getColours,
        );

        expect(
            [...colourMap.entries()],
        ).toEqual([
            [
                0,
                [1, 0, 0, 1],
            ],
            [
                1,
                [0, 1, 0, 1],
            ],
            [
                2,
                [0, 0, 1, 1],
            ],
        ]);
    });

    it("preserves existing colours", async () => {
        const existingColour = [
            0.25,
            0.5,
            0.75,
            1,
        ];

        const colourMap = new Map([
            [
                "tumour",
                existingColour,
            ],
        ]);

        const getColours =
            createColourGenerator();

        await assignAnnotationColours(
            colourMap,
            [
                "tumour",
                "stroma",
            ],
            getColours,
        );

        expect(
            colourMap.get("tumour"),
        ).toBe(existingColour);

        expect(
            colourMap.get("stroma"),
        ).toEqual([
            0,
            1,
            1,
            1,
        ]);

        expect(
            getColours,
        ).toHaveBeenCalledWith([
            "stroma",
        ]);
    });

    it("requests duplicate annotation types once", async () => {
        const colourMap = new Map();
        const getColours =
            createColourGenerator();

        await assignAnnotationColours(
            colourMap,
            [
                "tumour",
                "tumour",
            ],
            getColours,
        );

        expect(
            getColours,
        ).toHaveBeenCalledWith([
            "tumour",
        ]);

        expect(colourMap.size).toBe(1);
    });

    it("preserves numeric annotation type keys", async () => {
        const colourMap = new Map();
        const getColours =
            createColourGenerator();

        await assignAnnotationColours(
            colourMap,
            [0, 1],
            getColours,
        );

        expect(colourMap.has(0)).toBe(true);
        expect(colourMap.has("0")).toBe(false);
    });

    it("does not request colours for known types", async () => {
        const colourMap = new Map([
            [
                0,
                [1, 0, 0, 1],
            ],
        ]);

        const getColours =
            createColourGenerator();

        await assignAnnotationColours(
            colourMap,
            [0],
            getColours,
        );

        expect(
            getColours,
        ).not.toHaveBeenCalled();
    });

    it("uses configured colours before generated colours", async () => {
        const colourMap = new Map();
        const getColours =
            createColourGenerator();

        await assignAnnotationColours(
            colourMap,
            [
                "Tumour",
                "stroma",
            ],
            getColours,
            {
                Tumour: [
                    252,
                    161,
                    3,
                    255,
                ],
            },
        );

        expect(
            colourMap.get("Tumour"),
        ).toEqual([
            252 / 255,
            161 / 255,
            3 / 255,
            1,
        ]);

        expect(
            colourMap.get("stroma"),
        ).toEqual([
            0,
            1,
            1,
            1,
        ]);

        expect(
            getColours,
        ).toHaveBeenCalledExactlyOnceWith([
            "stroma",
        ]);
    });

    it("matches numeric annotation types to string config keys", async () => {
        const colourMap = new Map();
        const getColours =
            createColourGenerator();

        await assignAnnotationColours(
            colourMap,
            [2],
            getColours,
            {
                2: [
                    10,
                    20,
                    30,
                    255,
                ],
            },
        );

        expect(
            colourMap.get(2),
        ).toEqual([
            10 / 255,
            20 / 255,
            30 / 255,
            1,
        ]);

        expect(
            colourMap.has("2"),
        ).toBe(false);

        expect(
            getColours,
        ).not.toHaveBeenCalled();
    });
});
