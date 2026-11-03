import {
    afterEach,
    beforeEach,
    describe,
    expect,
    it,
    vi,
} from "vitest";

import {
    captureMapSnapshot,
    createSnapshotController,
    downloadSnapshot,
    downloadSnapshotMetadata,
} from "../../../tiatoolbox/visualization/openlayers/src/controls/snapshot.js";

function createMapHarness({
    size = [320, 240],
    targetElement = document.createElement("div"),
} = {}) {
    const originalTarget =
        document.createElement("div");

    let renderComplete = null;

    const map = {
        getSize: vi.fn(() => size),

        getTarget: vi.fn(
            () => originalTarget,
        ),

        getTargetElement: vi.fn(
            () => targetElement,
        ),

        once: vi.fn((event, callback) => {
            if (event === "rendercomplete") {
                renderComplete = callback;
            }
        }),

        setTarget: vi.fn((target) => {
            if (target !== originalTarget) {
                renderComplete?.();
            }
        }),

        updateSize: vi.fn(),
        renderSync: vi.fn(),
    };

    return {
        map,
        originalTarget,
        targetElement,
    };
}

function createScaleBar() {
    const scaleBar =
        document.createElement("div");

    const inner =
        document.createElement("div");

    inner.className =
        "ol-scale-line-inner";

    inner.textContent = "100 µm";

    scaleBar.appendChild(inner);

    scaleBar.style.backgroundColor =
        "rgba(17, 17, 17, 0.8)";

    inner.style.color = "#ffffff";
    inner.style.borderColor = "#ffffff";
    inner.style.borderStyle = "solid";
    inner.style.borderWidth = "1px";
    inner.style.fontSize = "10px";
    inner.style.fontFamily = "sans-serif";
    inner.style.fontWeight = "400";

    scaleBar.getBoundingClientRect =
        vi.fn(() => ({
            left: 20,
            top: 30,
            width: 110,
            height: 25,
        }));

    inner.getBoundingClientRect =
        vi.fn(() => ({
            left: 25,
            top: 35,
            width: 100,
            height: 15,
        }));

    return {
        scaleBar,
        inner,
    };
}

function readBlob(blob) {
    return new Promise(
        (resolve, reject) => {
            const reader =
                new FileReader();

            reader.addEventListener(
                "load",
                () => {
                    resolve(reader.result);
                },
            );

            reader.addEventListener(
                "error",
                () => {
                    reject(reader.error);
                },
            );

            reader.readAsText(blob);
        },
    );
}

beforeEach(() => {
    Object.defineProperty(
        URL,
        "createObjectURL",
        {
            configurable: true,
            value: vi.fn(
                () => "blob:snapshot",
            ),
        },
    );

    Object.defineProperty(
        URL,
        "revokeObjectURL",
        {
            configurable: true,
            value: vi.fn(),
        },
    );

    vi.spyOn(
        HTMLCanvasElement.prototype,
        "toBlob",
    ).mockImplementation(
        (callback) => {
            callback(
                new Blob(
                    ["snapshot"],
                    {
                        type: "image/png",
                    },
                ),
            );
        },
    );
});

afterEach(() => {
    vi.restoreAllMocks();
});

describe("snapshot capture", () => {
    it("captures the current map size as PNG", async () => {
        const {
            map,
            originalTarget,
        } = createMapHarness();

        const blob =
            await captureMapSnapshot(map);

        expect(blob).toBeInstanceOf(Blob);
        expect(blob.type).toBe("image/png");

        const snapshotCanvas =
            map.setTarget.mock.calls[0][0];

        expect(snapshotCanvas.width).toBe(320);
        expect(snapshotCanvas.height).toBe(240);

        expect(
            map.setTarget,
        ).toHaveBeenLastCalledWith(
            originalTarget,
        );

        expect(
            map.updateSize,
        ).toHaveBeenCalledOnce();

        expect(
            map.renderSync,
        ).toHaveBeenCalledOnce();
    });

    it("rejects capture when the map size is unavailable", async () => {
        const {
            map,
        } = createMapHarness();

        map.getSize.mockReturnValue(
            undefined,
        );

        await expect(
            captureMapSnapshot(map),
        ).rejects.toThrow(
            "The map size is not available for snapshot capture.",
        );
    });

    it("rejects capture when the map width is zero", async () => {
        const {
            map,
        } = createMapHarness({
            size: [0, 240],
        });

        await expect(
            captureMapSnapshot(map),
        ).rejects.toThrow(
            "The map size is not available for snapshot capture.",
        );
    });

    it("rejects capture when the map height is zero", async () => {
        const {
            map,
        } = createMapHarness({
            size: [320, 0],
        });

        await expect(
            captureMapSnapshot(map),
        ).rejects.toThrow(
            "The map size is not available for snapshot capture.",
        );
    });

    it("restores the map when PNG encoding fails", async () => {
        vi.spyOn(
            HTMLCanvasElement.prototype,
            "toBlob",
        ).mockImplementation(
            (callback) => {
                callback(null);
            },
        );

        const {
            map,
            originalTarget,
        } = createMapHarness();

        await expect(
            captureMapSnapshot(map),
        ).rejects.toThrow(
            "The snapshot could not be encoded as PNG.",
        );

        expect(
            map.setTarget,
        ).toHaveBeenLastCalledWith(
            originalTarget,
        );

        expect(
            map.updateSize,
        ).toHaveBeenCalledOnce();

        expect(
            map.renderSync,
        ).toHaveBeenCalledOnce();
    });

    it("draws a visible scale bar onto the snapshot", async () => {
        const targetElement =
            document.createElement("div");

        targetElement.getBoundingClientRect =
            vi.fn(() => ({
                left: 10,
                top: 20,
                width: 320,
                height: 240,
            }));

        const {
            scaleBar,
        } = createScaleBar();

        const context = {
            fillStyle: "",
            font: "",
            textAlign: "",
            textBaseline: "",
            fillRect: vi.fn(),
            fillText: vi.fn(),
        };

        vi.spyOn(
            HTMLCanvasElement.prototype,
            "getContext",
        ).mockReturnValue(context);

        const {
            map,
        } = createMapHarness({
            targetElement,
        });

        await captureMapSnapshot(
            map,
            scaleBar,
        );

        expect(
            context.fillRect,
        ).toHaveBeenCalled();

        expect(
            context.fillText,
        ).toHaveBeenCalledWith(
            "100 µm",
            65,
            22.5,
        );
    });

    it("does not draw a hidden scale bar", async () => {
        const {
            scaleBar,
        } = createScaleBar();

        scaleBar.classList.add(
            "viewer-control-hidden",
        );

        const getContext =
            vi.spyOn(
                HTMLCanvasElement.prototype,
                "getContext",
            );

        const {
            map,
        } = createMapHarness();

        await captureMapSnapshot(
            map,
            scaleBar,
        );

        expect(
            getContext,
        ).not.toHaveBeenCalled();
    });

    it("ignores a scale bar without an inner element", async () => {
        const scaleBar =
            document.createElement("div");

        const getContext =
            vi.spyOn(
                HTMLCanvasElement.prototype,
                "getContext",
            );

        const {
            map,
        } = createMapHarness();

        await captureMapSnapshot(
            map,
            scaleBar,
        );

        expect(
            getContext,
        ).not.toHaveBeenCalled();
    });

    it("continues when the canvas context is unavailable", async () => {
        const {
            scaleBar,
        } = createScaleBar();

        vi.spyOn(
            HTMLCanvasElement.prototype,
            "getContext",
        ).mockReturnValue(null);

        const {
            map,
        } = createMapHarness();

        await expect(
            captureMapSnapshot(
                map,
                scaleBar,
            ),
        ).resolves.toBeInstanceOf(Blob);
    });
});

describe("snapshot downloads", () => {
    it("downloads a snapshot using the supplied filename", () => {
        let clickedLink = null;

        vi.spyOn(
            HTMLAnchorElement.prototype,
            "click",
        ).mockImplementation(
            function click() {
                clickedLink = this;
            },
        );

        const blob =
            new Blob(
                ["snapshot"],
                {
                    type: "image/png",
                },
            );

        downloadSnapshot(
            blob,
            "slide_snapshot.png",
        );

        expect(
            URL.createObjectURL,
        ).toHaveBeenCalledWith(blob);

        expect(clickedLink.download).toBe(
            "slide_snapshot.png",
        );

        expect(clickedLink.href).toBe(
            "blob:snapshot",
        );

        expect(
            URL.revokeObjectURL,
        ).toHaveBeenCalledWith(
            "blob:snapshot",
        );
    });

    it("downloads formatted snapshot metadata", async () => {
        vi.spyOn(
            HTMLAnchorElement.prototype,
            "click",
        ).mockImplementation(
            () => {},
        );

        const metadata = {
            slide: {
                path: "slides/example.svs",
            },

            viewport: {
                zoom: 5,
            },
        };

        downloadSnapshotMetadata(
            metadata,
            "example_snapshot.json",
        );

        const blob =
            URL.createObjectURL.mock
                .calls[0][0];

        expect(blob).toBeInstanceOf(Blob);
        expect(blob.type).toBe(
            "application/json",
        );

        expect(
            await readBlob(blob),
        ).toBe(
            `${JSON.stringify(
                metadata,
                null,
                4,
            )}\n`,
        );
    });
});

describe("snapshot controller", () => {
    it("disables capture when no slide is loaded", async () => {
        const button =
            document.createElement("button");

        const onSnapshot = vi.fn();

        const controller =
            createSnapshotController({
                button,
                hasSlide: () => false,
                onSnapshot,
            });

        expect(button.disabled).toBe(true);

        expect(button.title).toBe(
            "Load a slide to save a snapshot",
        );

        await controller.saveSnapshot();

        expect(
            onSnapshot,
        ).not.toHaveBeenCalled();
    });

    it("disables capture while loading", async () => {
        const button =
            document.createElement("button");

        const onSnapshot = vi.fn();

        const controller =
            createSnapshotController({
                button,
                hasSlide: () => true,
                onSnapshot,
            });

        controller.setLoading(true);

        expect(button.disabled).toBe(true);

        expect(button.title).toBe(
            "Waiting for viewer to finish",
        );

        expect(button.innerHTML).toContain(
            "fa-spinner",
        );

        await controller.saveSnapshot();

        expect(
            onSnapshot,
        ).not.toHaveBeenCalled();
    });

    it("captures a snapshot when available", async () => {
        const button =
            document.createElement("button");

        const onSnapshot =
            vi.fn().mockResolvedValue();

        const controller =
            createSnapshotController({
                button,
                hasSlide: () => true,
                onSnapshot,
            });

        expect(button.disabled).toBe(false);

        await controller.saveSnapshot();

        expect(
            onSnapshot,
        ).toHaveBeenCalledOnce();

        expect(button.disabled).toBe(false);

        expect(button.title).toBe(
            "Save snapshot",
        );

        expect(button.innerHTML).toContain(
            "fa-camera",
        );
    });

    it("prevents another capture while saving", async () => {
        const button =
            document.createElement("button");

        let finishSnapshot;

        const onSnapshot =
            vi.fn(
                () =>
                    new Promise((resolve) => {
                        finishSnapshot =
                            resolve;
                    }),
            );

        const controller =
            createSnapshotController({
                button,
                hasSlide: () => true,
                onSnapshot,
            });

        const firstSnapshot =
            controller.saveSnapshot();

        expect(button.disabled).toBe(true);

        await controller.saveSnapshot();

        expect(
            onSnapshot,
        ).toHaveBeenCalledOnce();

        finishSnapshot();

        await firstSnapshot;

        expect(button.disabled).toBe(false);
    });

    it("restores the button when capture fails", async () => {
        const button =
            document.createElement("button");

        const controller =
            createSnapshotController({
                button,
                hasSlide: () => true,
                onSnapshot: vi
                    .fn()
                    .mockRejectedValue(
                        new Error(
                            "Capture failed",
                        ),
                    ),
            });

        await expect(
            controller.saveSnapshot(),
        ).rejects.toThrow(
            "Capture failed",
        );

        expect(button.disabled).toBe(false);

        expect(button.title).toBe(
            "Save snapshot",
        );
    });

    it("logs failed captures started from the button", async () => {
        const button =
            document.createElement("button");

        const error =
            new Error("Capture failed");

        const consoleError =
            vi.spyOn(
                console,
                "error",
            ).mockImplementation(
                () => {},
            );

        createSnapshotController({
            button,
            hasSlide: () => true,
            onSnapshot: vi
                .fn()
                .mockRejectedValue(error),
        });

        button.click();

        await vi.waitFor(() => {
            expect(
                consoleError,
            ).toHaveBeenCalledWith(
                "Failed to save snapshot.",
                error,
            );
        });
    });
});
