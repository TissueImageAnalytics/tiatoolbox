import {
    describe,
    expect,
    it,
    vi,
} from "vitest";

import {
    assignAnnotationColours,
    createAnnotationColourConfig,
    mergeAnnotationColourConfig,
    parseAnnotationColourConfig,
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
        // Test generated colours are assigned to new annotation types.
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
        // Test existing annotation colours are preserved.
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
        // Test duplicate annotation types request a generated colour only once.
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
        // Test numeric annotation type keys remain numeric.
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
        // Test known annotation types do not request new colours.
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
        // Test configured colours take priority over generated colours.
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
        // Test numeric annotation types match their string configuration keys.
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

describe("createAnnotationColourConfig", () => {
    it("exports loaded annotation colours as byte values", () => {
        // Test loaded annotation colours export as byte colour values.
        const colourMap = new Map([
            [
                "Tumour",
                [1, 0.5, 0, 0.4],
            ],
            [
                "Stroma",
                [0, 0.25, 1, 1],
            ],
        ]);

        expect(
            createAnnotationColourConfig(
                colourMap,
                [
                    "Tumour",
                    "Stroma",
                ],
            ),
        ).toEqual({
            color_dict: {
                Tumour: [
                    255,
                    128,
                    0,
                    255,
                ],
                Stroma: [
                    0,
                    64,
                    255,
                    255,
                ],
            },
        });
    });

    it("exports only currently loaded types", () => {
        // Test colour configuration exports only currently loaded annotation types.
        const colourMap = new Map([
            [
                "Tumour",
                [1, 0, 0, 1],
            ],
            [
                "Old type",
                [0, 1, 0, 1],
            ],
        ]);

        expect(
            createAnnotationColourConfig(
                colourMap,
                ["Tumour"],
            ),
        ).toEqual({
            color_dict: {
                Tumour: [
                    255,
                    0,
                    0,
                    255,
                ],
            },
        });
    });

    it("exports numeric annotation types as JSON-compatible keys", () => {
        // Test numeric annotation types export using JSON-compatible keys.
        const colourMap = new Map([
            [
                2,
                [0.1, 0.2, 0.3, 1],
            ],
        ]);

        expect(
            createAnnotationColourConfig(
                colourMap,
                [2],
            ),
        ).toEqual({
            color_dict: {
                2: [
                    26,
                    51,
                    77,
                    255,
                ],
            },
        });
    });

    it("skips loaded types without assigned colours", () => {
        // Test loaded annotation types without colours are not exported.
        const colourMap = new Map([
            [
                "Tumour",
                [1, 0, 0, 1],
            ],
        ]);

        expect(
            createAnnotationColourConfig(
                colourMap,
                [
                    "Tumour",
                    "Stroma",
                ],
            ),
        ).toEqual({
            color_dict: {
                Tumour: [
                    255,
                    0,
                    0,
                    255,
                ],
            },
        });
    });
});

describe("parseAnnotationColourConfig", () => {
    it("parses an old flat colour config", () => {
        const config =
            parseAnnotationColourConfig({
                color_dict: {
                    Tumour: [
                        255,
                        0,
                        0,
                        255,
                    ],
                },
            });

        expect(
            config,
        ).toEqual({
            colorDict: {
                Tumour: [
                    255,
                    0,
                    0,
                    255,
                ],
            },
            layerColorDicts: {},
        });
    });

    it("parses layer-specific colour config", () => {
        const config =
            parseAnnotationColourConfig({
                color_dict: {
                    Tumour: [
                        255,
                        0,
                        0,
                    ],
                },
                layer_color_dicts: {
                    nuclei: {
                        Tumour: [
                            0,
                            255,
                            0,
                            255,
                        ],
                    },
                },
            });

        expect(
            config.layerColorDicts,
        ).toEqual({
            nuclei: {
                Tumour: [
                    0,
                    255,
                    0,
                    255,
                ],
            },
        });
    });

    it("rejects configs without colour dictionaries", () => {
        expect(
            () =>
                parseAnnotationColourConfig(
                    {},
                ),
        ).toThrow(
            "Annotation colour config must contain color_dict or layer_color_dicts.",
        );
    });

    it("rejects invalid colour values", () => {
        expect(
            () =>
                parseAnnotationColourConfig({
                    color_dict: {
                        Tumour: [
                            256,
                            0,
                            0,
                        ],
                    },
                }),
        ).toThrow(
            "Invalid annotation colour for color_dict.Tumour.",
        );
    });

    it("rejects invalid layer colour dictionaries", () => {
        expect(
            () =>
                parseAnnotationColourConfig({
                    layer_color_dicts: {
                        nuclei:
                            "invalid",
                    },
                }),
        ).toThrow(
            "layer_color_dicts.nuclei must be an object.",
        );
    });

    it("rejects invalid config objects", () => {
        expect(
            () =>
                parseAnnotationColourConfig(
                    [],
                ),
        ).toThrow(
            "Annotation colour config must be an object.",
        );

        expect(
            () =>
                parseAnnotationColourConfig({
                    layer_color_dicts: [],
                }),
        ).toThrow(
            "layer_color_dicts must be an object.",
        );
    });
});


describe("mergeAnnotationColourConfig", () => {
    it("applies flat colours to matching annotation types", () => {
        const colours =
            new Map([
                [
                    "Tumour",
                    [
                        0,
                        0,
                        1,
                        0.4,
                    ],
                ],
                [
                    "Stroma",
                    [
                        0,
                        1,
                        0,
                        0.7,
                    ],
                ],
            ]);

        const result =
            mergeAnnotationColourConfig(
                colours,
                [
                    "Tumour",
                    "Stroma",
                ],
                {
                    Tumour: [
                        255,
                        128,
                        0,
                        255,
                    ],
                },
            );

        expect(
            result.get("Tumour"),
        ).toEqual([
            1,
            128 / 255,
            0,
            0.4,
        ]);

        expect(
            result.get("Stroma"),
        ).toEqual([
            0,
            1,
            0,
            0.7,
        ]);
    });

    it("prefers layer colours over flat colours", () => {
        const result =
            mergeAnnotationColourConfig(
                new Map([
                    [
                        "Tumour",
                        [
                            0,
                            0,
                            0,
                            0.5,
                        ],
                    ],
                ]),
                [
                    "Tumour",
                ],
                {
                    Tumour: [
                        255,
                        0,
                        0,
                    ],
                },
                {
                    Tumour: [
                        0,
                        255,
                        0,
                    ],
                },
            );

        expect(
            result.get("Tumour"),
        ).toEqual([
            0,
            1,
            0,
            0.5,
        ]);
    });

    it("supports numeric annotation type keys", () => {
        const result =
            mergeAnnotationColourConfig(
                new Map([
                    [
                        0,
                        [
                            0,
                            0,
                            0,
                            1,
                        ],
                    ],
                ]),
                [
                    0,
                ],
                {
                    0: [
                        64,
                        128,
                        255,
                    ],
                },
            );

        expect(
            result.get(0),
        ).toEqual([
            64 / 255,
            128 / 255,
            1,
            1,
        ]);
    });

    it("ignores colours for unloaded annotation types", () => {
        const original =
            new Map([
                [
                    "Tumour",
                    [
                        1,
                        0,
                        0,
                        0.6,
                    ],
                ],
            ]);

        const result =
            mergeAnnotationColourConfig(
                original,
                [
                    "Tumour",
                ],
                {
                    Missing: [
                        0,
                        255,
                        0,
                    ],
                },
            );

        expect(
            result,
        ).toEqual(
            original,
        );
    });
});
