// Capture the visible scale bar state before moving the map target.
function getScaleBarSnapshot(map, scaleBarElement) {
    if (
        scaleBarElement === null ||
        scaleBarElement.classList.contains(
            "viewer-control-hidden",
        )
    ) {
        return null;
    }

    const targetElement = map.getTargetElement();
    const inner =
        scaleBarElement.querySelector(
            ".ol-scale-line-inner",
        );

    if (
        targetElement === null ||
        inner === null
    ) {
        return null;
    }

    const targetRect =
        targetElement.getBoundingClientRect();

    const outerRect =
        scaleBarElement.getBoundingClientRect();

    const innerRect =
        inner.getBoundingClientRect();

    const outerStyle =
        window.getComputedStyle(
            scaleBarElement,
        );

    const innerStyle =
        window.getComputedStyle(inner);

    return {
        x: outerRect.left - targetRect.left,
        y: outerRect.top - targetRect.top,
        width: outerRect.width,
        height: outerRect.height,
        backgroundColour:
            outerStyle.backgroundColor,

        inner: {
            x: innerRect.left - targetRect.left,
            y: innerRect.top - targetRect.top,
            width: innerRect.width,
            height: innerRect.height,
            text: inner.textContent.trim(),
            colour: innerStyle.color,
            font:
                `${innerStyle.fontWeight} ` +
                `${innerStyle.fontSize} ` +
                `${innerStyle.fontFamily}`,
            borderTopWidth:
                Number.parseFloat(
                    innerStyle.borderTopWidth,
                ) || 0,
            borderRightWidth:
                Number.parseFloat(
                    innerStyle.borderRightWidth,
                ) || 0,
            borderBottomWidth:
                Number.parseFloat(
                    innerStyle.borderBottomWidth,
                ) || 0,
            borderLeftWidth:
                Number.parseFloat(
                    innerStyle.borderLeftWidth,
                ) || 0,
            borderTopColour:
                innerStyle.borderTopColor,
            borderRightColour:
                innerStyle.borderRightColor,
            borderBottomColour:
                innerStyle.borderBottomColor,
            borderLeftColour:
                innerStyle.borderLeftColor,
        },
    };
}

// Draw the captured scale bar onto the map canvas.
function drawScaleBar(canvas, scaleBar) {
    if (scaleBar === null) {
        return;
    }

    const context =
        canvas.getContext("2d");

    if (context === null) {
        return;
    }

    context.fillStyle =
        scaleBar.backgroundColour;

    context.fillRect(
        scaleBar.x,
        scaleBar.y,
        scaleBar.width,
        scaleBar.height,
    );

    const inner = scaleBar.inner;

    context.fillStyle =
        inner.borderTopColour;

    context.fillRect(
        inner.x,
        inner.y,
        inner.width,
        inner.borderTopWidth,
    );

    context.fillStyle =
        inner.borderRightColour;

    context.fillRect(
        inner.x +
            inner.width -
            inner.borderRightWidth,
        inner.y,
        inner.borderRightWidth,
        inner.height,
    );

    context.fillStyle =
        inner.borderBottomColour;

    context.fillRect(
        inner.x,
        inner.y +
            inner.height -
            inner.borderBottomWidth,
        inner.width,
        inner.borderBottomWidth,
    );

    context.fillStyle =
        inner.borderLeftColour;

    context.fillRect(
        inner.x,
        inner.y,
        inner.borderLeftWidth,
        inner.height,
    );

    context.fillStyle = inner.colour;
    context.font = inner.font;
    context.textAlign = "center";
    context.textBaseline = "middle";

    context.fillText(
        inner.text,
        inner.x + inner.width / 2,
        inner.y + inner.height / 2,
    );
}

// Render the current map viewport to a PNG blob.
async function captureMapSnapshot(
    map,
    scaleBarElement = null,
) {
    const size = map.getSize();

    if (
        size === undefined ||
        size[0] <= 0 ||
        size[1] <= 0
    ) {
        throw new Error(
            "The map size is not available for snapshot capture.",
        );
    }

    const scaleBar =
        getScaleBarSnapshot(
            map,
            scaleBarElement,
        );

    const originalTarget = map.getTarget();
    const canvas = document.createElement("canvas");

    canvas.width = size[0];
    canvas.height = size[1];

    const renderComplete = new Promise((resolve) => {
        map.once("rendercomplete", resolve);
    });

    map.setTarget(canvas);

    try {
        await renderComplete;

        drawScaleBar(
            canvas,
            scaleBar,
        );

        const blob = await new Promise(
            (resolve, reject) => {
                canvas.toBlob((result) => {
                    if (result === null) {
                        reject(
                            new Error(
                                "The snapshot could not be encoded as PNG.",
                            ),
                        );
                        return;
                    }

                    resolve(result);
                }, "image/png");
            },
        );

        return blob;
    } finally {
        map.setTarget(originalTarget);
        map.updateSize();
        map.renderSync();
    }
}

// Download a snapshot blob using the supplied filename.
function downloadSnapshot(blob, filename) {
    const url = URL.createObjectURL(blob);
    const downloadLink =
        document.createElement("a");

    downloadLink.href = url;
    downloadLink.download = filename;

    document.body.appendChild(
        downloadLink,
    );

    downloadLink.click();
    downloadLink.remove();

    URL.revokeObjectURL(url);
}

// Download snapshot metadata as formatted JSON.
function downloadSnapshotMetadata(
    metadata,
    filename,
) {
    const json = `${JSON.stringify(
        metadata,
        null,
        4,
    )}\n`;

    const blob = new Blob(
        [json],
        {
            type: "application/json",
        },
    );

    downloadSnapshot(
        blob,
        filename,
    );
}

const snapshotAnnotationDisplayModes =
    new Set([
        "type",
        "property",
        "secondary",
    ]);

const snapshotAnnotationColourMaps =
    new Set([
        "viridis",
        "plasma",
        "inferno",
        "magma",
        "cividis",
        "turbo",
    ]);

function isSnapshotAnnotationType(value) {
    return (
        value === null ||
        typeof value === "string" ||
        typeof value === "boolean" ||
        Number.isFinite(value)
    );
}

function isSnapshotAnnotationColour(colour) {
    return (
        Array.isArray(colour) &&
        colour.length === 4 &&
        colour.every(
            (channel) =>
                Number.isFinite(channel) &&
                channel >= 0 &&
                channel <= 1,
        )
    );
}

// Parse and validate snapshot metadata.
function parseSnapshotMetadata(json) {
    let metadata;

    try {
        metadata = JSON.parse(json);
    } catch {
        throw new Error(
            "Snapshot metadata is not valid JSON.",
        );
    }

    if (
        metadata === null ||
        typeof metadata !== "object" ||
        Array.isArray(metadata)
    ) {
        throw new Error(
            "Snapshot metadata must be a JSON object.",
        );
    }

    if (metadata.version !== 1) {
        throw new Error(
            "Unsupported snapshot metadata version.",
        );
    }

    if (
        metadata.slide === null ||
        typeof metadata.slide !== "object" ||
        typeof metadata.slide.path !== "string" ||
        metadata.slide.path.length === 0
    ) {
        throw new Error(
            "Snapshot metadata does not contain a valid slide path.",
        );
    }

    const viewport = metadata.viewport;

    if (
        viewport === null ||
        typeof viewport !== "object" ||
        !Array.isArray(viewport.centre) ||
        viewport.centre.length !== 2 ||
        !viewport.centre.every(Number.isFinite) ||
        !Number.isFinite(viewport.zoom) ||
        !Number.isFinite(viewport.rotation)
    ) {
        throw new Error(
            "Snapshot metadata does not contain a valid viewport.",
        );
    }

    if (
        !Array.isArray(metadata.overlays) ||
        !metadata.overlays.every((overlay) =>
            overlay !== null &&
            typeof overlay === "object" &&
            typeof overlay.name === "string" &&
            overlay.name.length > 0 &&
            typeof overlay.path === "string" &&
            overlay.path.length > 0 &&
            typeof overlay.visible === "boolean" &&
            Number.isFinite(overlay.opacity) &&
            overlay.opacity >= 0 &&
            overlay.opacity <= 1 &&
            (
                overlay.zIndex === null ||
                Number.isFinite(overlay.zIndex)
            ) &&
            Number.isInteger(overlay.order) &&
            overlay.order >= 0
        )
    ) {
        throw new Error(
            "Snapshot metadata contains an invalid overlay.",
        );
    }

    const scaleBar = metadata.scaleBar;

    if (
        scaleBar === null ||
        typeof scaleBar !== "object" ||
        typeof scaleBar.enabled !== "boolean" ||
        typeof scaleBar.included !== "boolean" ||
        typeof scaleBar.colour !== "string" ||
        !/^#[0-9a-fA-F]{6}$/.test(
            scaleBar.colour,
        ) ||
        !Number.isFinite(scaleBar.opacity) ||
        scaleBar.opacity < 0 ||
        scaleBar.opacity > 100 ||
        !["small", "default", "large"].includes(
            scaleBar.size,
        ) ||
        !["metric", "imperial"].includes(
            scaleBar.units,
        )
    ) {
        throw new Error(
            "Snapshot metadata does not contain valid scale bar settings.",
        );
    }

    const annotations =
        metadata.annotations;

    if (
        annotations === null ||
        typeof annotations !== "object" ||
        Array.isArray(annotations) ||
        !snapshotAnnotationDisplayModes.has(
            annotations.displayMode,
        ) ||
        !snapshotAnnotationColourMaps.has(
            annotations.colourMap,
        ) ||
        !Array.isArray(annotations.layers)
    ) {
        throw new Error(
            "Snapshot metadata does not contain valid annotation settings.",
        );
    }

    if (
        (
            annotations.displayMode === "type" &&
            (
                annotations.property !== null ||
                annotations.secondarySelection !== null
            )
        ) ||
        (
            annotations.displayMode === "property" &&
            (
                typeof annotations.property !== "string" ||
                annotations.property.length === 0 ||
                annotations.secondarySelection !== null
            )
        )
    ) {
        throw new Error(
            "Snapshot metadata does not contain valid annotation settings.",
        );
    }

    const secondarySelection =
        annotations.secondarySelection;

    if (
        annotations.displayMode === "secondary" &&
        (
            typeof annotations.property !== "string" ||
            annotations.property.length === 0 ||
            secondarySelection === null ||
            typeof secondarySelection !== "object" ||
            Array.isArray(secondarySelection) ||
            typeof secondarySelection.layerName !==
                "string" ||
            secondarySelection.layerName.length === 0 ||
            !isSnapshotAnnotationType(
                secondarySelection.annotationType,
            )
        )
    ) {
        throw new Error(
            "Snapshot metadata does not contain valid annotation settings.",
        );
    }

    const overlayNames =
        new Set(
            metadata.overlays.map(
                (overlay) => overlay.name,
            ),
        );

    if (
        !annotations.layers.every(
            (layer) =>
                layer !== null &&
                typeof layer === "object" &&
                !Array.isArray(layer) &&
                typeof layer.name === "string" &&
                layer.name.length > 0 &&
                overlayNames.has(layer.name) &&
                Array.isArray(layer.types) &&
                layer.types.every(
                    (annotation) =>
                        annotation !== null &&
                        typeof annotation ===
                            "object" &&
                        !Array.isArray(
                            annotation,
                        ) &&
                        isSnapshotAnnotationType(
                            annotation.annotationType,
                        ) &&
                        isSnapshotAnnotationColour(
                            annotation.colour,
                        ) &&
                        typeof annotation.visible ===
                            "boolean" &&
                        Number.isFinite(
                            annotation.opacity,
                        ) &&
                        annotation.opacity >= 0 &&
                        annotation.opacity <= 1,
                ),
        )
    ) {
        throw new Error(
            "Snapshot metadata contains invalid annotation layer state.",
        );
    }

    if (
        annotations.displayMode === "secondary" &&
        !annotations.layers.some(
            (layer) =>
                layer.name ===
                    secondarySelection.layerName &&
                layer.types.some(
                    (annotation) =>
                        Object.is(
                            annotation.annotationType,
                            secondarySelection.annotationType,
                        ),
                ),
        )
    ) {
        throw new Error(
            "Snapshot metadata contains an invalid secondary annotation selection.",
        );
    }

    return metadata;
}

// Manage snapshot availability and capture.
function createSnapshotController({
    button,
    hasSlide,
    onSnapshot,
}) {
    let loading = false;
    let saving = false;

    function updateButton() {
        const slideLoaded = hasSlide();
        const busy = loading || saving;

        button.disabled =
            busy || !slideLoaded;

        button.innerHTML = busy
            ? '<i class="fas fa-spinner fa-spin"></i>'
            : '<i class="fas fa-camera"></i>';

        button.title = !slideLoaded
            ? "Load a slide to save a snapshot"
            : (
                busy
                    ? "Waiting for viewer to finish"
                    : "Save snapshot"
            );
    }

    function setLoading(value) {
        loading = value;
        updateButton();
    }

    async function saveSnapshot() {
        if (
            loading ||
            saving ||
            !hasSlide()
        ) {
            return;
        }

        saving = true;
        updateButton();

        try {
            await onSnapshot();
        } finally {
            saving = false;
            updateButton();
        }
    }

    button.addEventListener("click", () => {
        saveSnapshot().catch((error) => {
            console.error(
                "Failed to save snapshot.",
                error,
            );
        });
    });

    updateButton();

    return {
        saveSnapshot,
        setLoading,
        updateButton,
    };
}

export {
    captureMapSnapshot,
    createSnapshotController,
    downloadSnapshot,
    downloadSnapshotMetadata,
    parseSnapshotMetadata,
};
