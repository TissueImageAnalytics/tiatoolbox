import {
    afterEach,
    describe,
    expect,
    it,
    vi,
} from "vitest";

import {
    clearOverlays,
    createSession,
    getConfiguredFiles,
    loadOverlay,
    loadSlide,
    removeOverlay,
    removeSlide,
    getAnnotationColors,
    getAnnotationProperties,
    getAnnotationPropertyValues,
    setAnnotationColors,
    setAnnotationFilter,
    setAnnotationMapper,
    setAnnotationProperty,
    setAnnotationPropertyRange,
} from "../../../tiatoolbox/visualization/openlayers/src/api/tileserver.js";

function mockResponse({
    ok = true,
    json,
} = {}) {
    return {
        ok,
        json: vi.fn().mockResolvedValue(json),
    };
}

afterEach(() => {
    vi.unstubAllGlobals();
});

describe("createSession", () => {
    it("creates a TileServer session", async () => {
        // Test creating a TileServer session.
        const fetchMock = vi.fn().mockResolvedValue(
            mockResponse({
                json: {
                    session_id: "test-session",
                },
            }),
        );

        vi.stubGlobal("fetch", fetchMock);

        await expect(
            createSession(),
        ).resolves.toBe("test-session");

        expect(
            fetchMock,
        ).toHaveBeenCalledExactlyOnceWith(
            "/tileserver/session_id",
        );
    });

    it("throws when the session cannot be created", async () => {
        // Test error handling when a TileServer session cannot be created.
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue(
                mockResponse({
                    ok: false,
                }),
            ),
        );

        await expect(
            createSession(),
        ).rejects.toThrow(
            "Failed to create TileServer session.",
        );
    });
});

describe("loadSlide", () => {
    it("loads a slide and returns its metadata", async () => {
        // Test loading a slide and returning its metadata.
        const slideMetadata = {
            width: 1000,
            height: 800,
        };

        const fetchMock = vi
            .fn()
            .mockResolvedValueOnce(
                mockResponse(),
            )
            .mockResolvedValueOnce(
                mockResponse({
                    json: slideMetadata,
                }),
            );

        vi.stubGlobal("fetch", fetchMock);

        await expect(
            loadSlide("/slides/CMU-1.svs"),
        ).resolves.toEqual(slideMetadata);

        expect(fetchMock).toHaveBeenCalledTimes(2);

        const [
            loadUrl,
            loadOptions,
        ] = fetchMock.mock.calls[0];

        expect(loadUrl).toBe(
            "/tileserver/slide",
        );
        expect(loadOptions.method).toBe("PUT");
        expect(
            loadOptions.body,
        ).toBeInstanceOf(FormData);
        expect(
            loadOptions.body.get("slide_path"),
        ).toBe("/slides/CMU-1.svs");

        expect(fetchMock.mock.calls[1]).toEqual([
            "/tileserver/slide",
        ]);
    });

    it("throws when the slide cannot be loaded", async () => {
        // Test error handling when a slide cannot be loaded.
        const fetchMock = vi.fn().mockResolvedValue(
            mockResponse({
                ok: false,
            }),
        );

        vi.stubGlobal("fetch", fetchMock);

        await expect(
            loadSlide("/slides/CMU-1.svs"),
        ).rejects.toThrow(
            "Failed to load slide: /slides/CMU-1.svs",
        );

        expect(fetchMock).toHaveBeenCalledOnce();
    });

    it("throws when slide metadata cannot be retrieved", async () => {
        // Test error handling when slide metadata cannot be retrieved.
        const fetchMock = vi
            .fn()
            .mockResolvedValueOnce(
                mockResponse(),
            )
            .mockResolvedValueOnce(
                mockResponse({
                    ok: false,
                }),
            );

        vi.stubGlobal("fetch", fetchMock);

        await expect(
            loadSlide("/slides/CMU-1.svs"),
        ).rejects.toThrow(
            "Failed to retrieve slide metadata.",
        );

        expect(fetchMock).toHaveBeenCalledTimes(2);
    });
});

describe("getConfiguredFiles", () => {
    it("returns configured slide files", async () => {
        // Test returning configured slide files.
        const files = {
            directory: "slides",
            files: [
                {
                    name: "CMU-1.svs",
                    path: "slides/CMU-1.svs",
                },
                {
                    name: "CMU-2.svs",
                    path: "slides/CMU-2.svs",
                },
            ],
        };

        const fetchMock = vi.fn().mockResolvedValue(
            mockResponse({
                json: files,
            }),
        );

        vi.stubGlobal("fetch", fetchMock);

        await expect(
            getConfiguredFiles("slide"),
        ).resolves.toEqual(files);

        expect(fetchMock).toHaveBeenCalledWith(
            "/tileserver/files/slide",
        );
    });

    it("throws when configured overlay files cannot be retrieved", async () => {
        // Test error handling when configured overlay files cannot be retrieved.
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue(
                mockResponse({
                    ok: false,
                }),
            ),
        );

        await expect(
            getConfiguredFiles("overlay"),
        ).rejects.toThrow(
            "Failed to get configured overlay files.",
        );
    });
});

describe("clearOverlays", () => {
    it("clears all overlays", async () => {
        // Test clearing all overlays.
        const fetchMock = vi.fn().mockResolvedValue(
            mockResponse(),
        );

        vi.stubGlobal("fetch", fetchMock);

        await expect(
            clearOverlays(),
        ).resolves.toBeUndefined();

        expect(fetchMock).toHaveBeenCalledWith(
            "/tileserver/clear_overlays",
            {
                method: "PUT",
            },
        );
    });

    it("throws when overlays cannot be cleared", async () => {
        // Test error handling when overlays cannot be cleared.
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue(
                mockResponse({
                    ok: false,
                }),
            ),
        );

        await expect(
            clearOverlays(),
        ).rejects.toThrow(
            "Failed to clear overlays.",
        );
    });
});

describe("removeSlide", () => {
    it("removes the current slide", async () => {
        // Test removing the current slide.
        const fetchMock = vi.fn().mockResolvedValue(
            mockResponse(),
        );

        vi.stubGlobal("fetch", fetchMock);

        await expect(
            removeSlide(),
        ).resolves.toBeUndefined();

        expect(fetchMock).toHaveBeenCalledWith(
            "/tileserver/slide",
            {
                method: "DELETE",
            },
        );
    });

    it("throws when the current slide cannot be removed", async () => {
        // Test error handling when the current slide cannot be removed.
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue(
                mockResponse({
                    ok: false,
                }),
            ),
        );

        await expect(
            removeSlide(),
        ).rejects.toThrow(
            "Failed to remove the current slide.",
        );
    });
});

describe("loadOverlay", () => {
    it("loads an overlay and returns its metadata", async () => {
        // Test loading an overlay and returning its metadata.
        const overlayMetadata = {
            layer: "Tumour",
        };

        const fetchMock = vi.fn().mockResolvedValue(
            mockResponse({
                json: overlayMetadata,
            }),
        );

        vi.stubGlobal("fetch", fetchMock);

        await expect(
            loadOverlay(
                "/overlays/tumour.db",
                "Tumour",
            ),
        ).resolves.toEqual(overlayMetadata);

        expect(fetchMock).toHaveBeenCalledOnce();

        const [
            url,
            options,
        ] = fetchMock.mock.calls[0];

        expect(url).toBe(
            "/tileserver/overlay",
        );
        expect(options.method).toBe("PUT");
        expect(
            options.body,
        ).toBeInstanceOf(FormData);
        expect(
            options.body.get("overlay_path"),
        ).toBe("/overlays/tumour.db");
        expect(
            options.body.get("layer_name"),
        ).toBe("Tumour");
    });

    it("throws when an overlay cannot be loaded", async () => {
        // Test error handling when an overlay cannot be loaded.
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue(
                mockResponse({
                    ok: false,
                }),
            ),
        );

        await expect(
            loadOverlay(
                "/overlays/tumour.db",
                "Tumour",
            ),
        ).rejects.toThrow(
            "Failed to load overlay: /overlays/tumour.db",
        );
    });
});

describe("removeOverlay", () => {
    it("removes an overlay using its encoded layer name", async () => {
        // Test encoding the layer name when removing an overlay.
        const fetchMock = vi.fn().mockResolvedValue(
            mockResponse(),
        );

        vi.stubGlobal("fetch", fetchMock);

        await expect(
            removeOverlay("Tumour / Stroma"),
        ).resolves.toBeUndefined();

        expect(fetchMock).toHaveBeenCalledWith(
            "/tileserver/overlay/Tumour%20%2F%20Stroma",
            {
                method: "DELETE",
            },
        );
    });

    it("throws when an overlay cannot be removed", async () => {
        // Test error handling when an overlay cannot be removed.
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue(
                mockResponse({
                    ok: false,
                }),
            ),
        );

        await expect(
            removeOverlay("Tumour"),
        ).rejects.toThrow(
            "Failed to remove overlay: Tumour",
        );
    });
});

describe("setAnnotationFilter", () => {
    it("sends an annotation filter to TileServer", async () => {
        // Test sending an annotation visibility filter.
        const fetchMock = vi.fn().mockResolvedValue(
            mockResponse(),
        );

        vi.stubGlobal("fetch", fetchMock);

        const where =
            'props["type"]=="Tumour"';

        await expect(
            setAnnotationFilter(where),
        ).resolves.toBeUndefined();

        expect(fetchMock).toHaveBeenCalledOnce();

        const [
            url,
            options,
        ] = fetchMock.mock.calls[0];

        expect(url).toBe(
            "/tileserver/renderer/where",
        );
        expect(options.method).toBe("PUT");
        expect(
            options.body,
        ).toBeInstanceOf(FormData);

        expect(
            options.body.get("val"),
        ).toBe(
            JSON.stringify(where),
        );
    });

    it("clears the annotation filter", async () => {
        // Test clearing the annotation visibility filter.
        const fetchMock = vi.fn().mockResolvedValue(
            mockResponse(),
        );

        vi.stubGlobal("fetch", fetchMock);

        await expect(
            setAnnotationFilter(null),
        ).resolves.toBeUndefined();

        const [
            ,
            options,
        ] = fetchMock.mock.calls[0];

        expect(
            options.body.get("val"),
        ).toBe("null");
    });

    it("throws when the annotation filter cannot be updated", async () => {
        // Test error handling when annotation visibility cannot be updated.
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue(
                mockResponse({
                    ok: false,
                }),
            ),
        );

        await expect(
            setAnnotationFilter(null),
        ).rejects.toThrow(
            "Failed to update annotation visibility.",
        );
    });
});

describe("setAnnotationColors", () => {
    it("sends annotation colours to TileServer", async () => {
        // Test sending annotation colours to TileServer in the expected format.
        const fetchMock = vi.fn().mockResolvedValue(
            mockResponse(),
        );

        vi.stubGlobal("fetch", fetchMock);

        const colorMap = {
            Tumour: "#ff0000",
            Stroma: "#00ff00",
        };

        await expect(
            setAnnotationColors(colorMap),
        ).resolves.toBeUndefined();

        expect(fetchMock).toHaveBeenCalledOnce();

        const [
            url,
            options,
        ] = fetchMock.mock.calls[0];

        expect(url).toBe(
            "/tileserver/cmap",
        );
        expect(options.method).toBe("PUT");
        expect(
            options.body,
        ).toBeInstanceOf(FormData);

        expect(
            JSON.parse(
                options.body.get("cmap"),
            ),
        ).toEqual({
            keys: [
                "Tumour",
                "Stroma",
            ],
            values: [
                "#ff0000",
                "#00ff00",
            ],
        });
    });

    it("preserves numeric annotation type keys from a map", async () => {
        const fetchMock = vi.fn().mockResolvedValue(
            mockResponse(),
        );

        vi.stubGlobal("fetch", fetchMock);

        const colorMap = new Map([
            [
                0,
                [1, 0, 0, 1],
            ],
            [
                1,
                [0, 1, 0, 1],
            ],
        ]);

        await setAnnotationColors(colorMap);

        const [
            ,
            options,
        ] = fetchMock.mock.calls[0];

        expect(
            JSON.parse(
                options.body.get("cmap"),
            ),
        ).toEqual({
            keys: [
                0,
                1,
            ],
            values: [
                [1, 0, 0, 1],
                [0, 1, 0, 1],
            ],
        });
    });

    it("throws when annotation colours cannot be updated", async () => {
        // Test error handling when annotation colours cannot be updated.
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue(
                mockResponse({
                    ok: false,
                }),
            ),
        );

        await expect(
            setAnnotationColors({
                Tumour: "#ff0000",
            }),
        ).rejects.toThrow(
            "Failed to update annotation colours.",
        );
    });

    it("gets annotation colours while preserving type keys", async () => {
        const fetchMock = vi.fn().mockResolvedValue({
            ok: true,

            json: vi.fn().mockResolvedValue({
                keys: [
                    0,
                    1,
                ],

                values: [
                    [1, 0, 0, 1],
                    [0, 1, 0, 1],
                ],
            }),
        });

        vi.stubGlobal(
            "fetch",
            fetchMock,
        );

        const colourMap =
            await getAnnotationColors([
                0,
                1,
            ]);

        const [
            url,
            options,
        ] = fetchMock.mock.calls[0];

        expect(url).toBe(
            "/tileserver/annotation_colours",
        );

        expect(options.method).toBe("PUT");

        expect(
            JSON.parse(
                options.body.get("types"),
            ),
        ).toEqual([
            0,
            1,
        ]);

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
        ]);
    });
});

describe("annotation properties", () => {
    it("gets annotation properties for a layer", async () => {
        // Test annotation properties are requested for the selected layer.
        const fetchMock = vi.fn().mockResolvedValue(
            mockResponse({
                json: [
                    "type",
                    "prob",
                ],
            }),
        );

        vi.stubGlobal("fetch", fetchMock);

        const properties =
            await getAnnotationProperties(
                "semantic_segmentation",
            );

        expect(
            fetchMock,
        ).toHaveBeenCalledExactlyOnceWith(
            "/tileserver/prop_names/all?layer=semantic_segmentation",
        );

        expect(properties).toEqual([
            "type",
            "prob",
        ]);
    });

    it("gets annotation property values for a layer", async () => {
        // Test property values are requested for the selected annotation layer.
        const fetchMock = vi.fn().mockResolvedValue(
            mockResponse({
                json: [
                    0.2,
                    0.8,
                ],
            }),
        );

        vi.stubGlobal("fetch", fetchMock);

        const values =
            await getAnnotationPropertyValues(
                "nucleus_detection",
                "prob",
            );

        expect(
            fetchMock,
        ).toHaveBeenCalledExactlyOnceWith(
            "/tileserver/prop_values/prob/all?layer=nucleus_detection",
        );

        expect(values).toEqual([
            0.2,
            0.8,
        ]);
    });

    it("rejects failed annotation property requests", async () => {
        // Test a failed annotation property request is rejected.
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue(
                mockResponse({
                    ok: false,
                }),
            ),
        );

        await expect(
            getAnnotationProperties(
                "annotations",
            ),
        ).rejects.toThrow(
            "Failed to get annotation properties.",
        );
    });

    it("rejects failed annotation property value requests", async () => {
        // Test a failed annotation property value request is rejected.
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue(
                mockResponse({
                    ok: false,
                }),
            ),
        );

        await expect(
            getAnnotationPropertyValues(
                "annotations",
                "prob",
            ),
        ).rejects.toThrow(
            "Failed to get annotation property values.",
        );
    });
});

describe("annotation display", () => {
    it("sets the annotation property", async () => {
        // Test changing the property used to colour annotations.
        const fetchMock = vi.fn().mockResolvedValue(
            mockResponse(),
        );

        vi.stubGlobal("fetch", fetchMock);

        await expect(
            setAnnotationProperty("prob"),
        ).resolves.toBeUndefined();

        const [
            url,
            options,
        ] = fetchMock.mock.calls[0];

        expect(url).toBe(
            "/tileserver/renderer/score_prop",
        );

        expect(options.method).toBe("PUT");

        expect(
            options.body.get("val"),
        ).toBe(
            JSON.stringify("prob"),
        );
    });

    it("sets the annotation mapper", async () => {
        // Test changing the annotation colour mapper.
        const fetchMock = vi.fn().mockResolvedValue(
            mockResponse(),
        );

        vi.stubGlobal("fetch", fetchMock);

        await expect(
            setAnnotationMapper("viridis"),
        ).resolves.toBeUndefined();

        const [
            url,
            options,
        ] = fetchMock.mock.calls[0];

        expect(url).toBe(
            "/tileserver/cmap",
        );

        expect(options.method).toBe("PUT");

        expect(
            options.body.get("cmap"),
        ).toBe(
            JSON.stringify("viridis"),
        );
    });

    it("sets the annotation property range", async () => {
        // Test changing the range used by the annotation mapper.
        const fetchMock = vi.fn().mockResolvedValue(
            mockResponse(),
        );

        vi.stubGlobal("fetch", fetchMock);

        await expect(
            setAnnotationPropertyRange([
                0.2,
                0.8,
            ]),
        ).resolves.toBeUndefined();

        const [
            url,
            options,
        ] = fetchMock.mock.calls[0];

        expect(url).toBe(
            "/tileserver/prop_range",
        );

        expect(options.method).toBe("PUT");

        expect(
            options.body.get("range"),
        ).toBe(
            JSON.stringify([
                0.2,
                0.8,
            ]),
        );
    });

    it("rejects a failed annotation property update", async () => {
        // Test error handling when the annotation property cannot be updated.
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue(
                mockResponse({
                    ok: false,
                }),
            ),
        );

        await expect(
            setAnnotationProperty("prob"),
        ).rejects.toThrow(
            "Failed to update annotation property.",
        );
    });

    it("rejects a failed annotation mapper update", async () => {
        // Test error handling when the annotation mapper cannot be updated.
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue(
                mockResponse({
                    ok: false,
                }),
            ),
        );

        await expect(
            setAnnotationMapper("viridis"),
        ).rejects.toThrow(
            "Failed to update annotation colour map.",
        );
    });

    it("rejects a failed annotation property range update", async () => {
        // Test error handling when the annotation property range cannot be updated.
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue(
                mockResponse({
                    ok: false,
                }),
            ),
        );

        await expect(
            setAnnotationPropertyRange([
                0,
                1,
            ]),
        ).rejects.toThrow(
            "Failed to update annotation property range.",
        );
    });
});
