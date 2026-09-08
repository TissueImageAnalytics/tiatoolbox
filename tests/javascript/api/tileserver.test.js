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
    setAnnotationColors,
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

        expect(fetchMock).toHaveBeenCalledOnce();
        expect(fetchMock).toHaveBeenCalledWith(
            "/tileserver/session_id",
        );
    });

    it("throws when the session cannot be created", async () => {
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
    it("returns configured files", async () => {
        const files = [
            "/slides/CMU-1.svs",
            "/slides/CMU-2.svs",
        ];

        const fetchMock = vi.fn().mockResolvedValue(
            mockResponse({
                json: files,
            }),
        );

        vi.stubGlobal("fetch", fetchMock);

        await expect(
            getConfiguredFiles("slides"),
        ).resolves.toEqual(files);

        expect(fetchMock).toHaveBeenCalledWith(
            "/tileserver/files/slides",
        );
    });

    it("throws when configured files cannot be retrieved", async () => {
        vi.stubGlobal(
            "fetch",
            vi.fn().mockResolvedValue(
                mockResponse({
                    ok: false,
                }),
            ),
        );

        await expect(
            getConfiguredFiles("overlays"),
        ).rejects.toThrow(
            "Failed to get configured overlays files.",
        );
    });
});

describe("clearOverlays", () => {
    it("clears all overlays", async () => {
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

describe("setAnnotationColors", () => {
    it("sends annotation colours to TileServer", async () => {
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

    it("throws when annotation colours cannot be updated", async () => {
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
});
