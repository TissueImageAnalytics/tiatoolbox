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
};
