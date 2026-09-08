// Initialise the TileServer session used for dynamic slide loading.
async function createSession() {
    const response = await fetch("/tileserver/session_id");

    if (!response.ok) {
        throw new Error("Failed to create TileServer session.");
    }

    const data = await response.json();

    return data.session_id;
}

// Load a slide into the current TileServer session and return its metadata.
async function loadSlide(slidePath) {
    const formData = new FormData();
    formData.append("slide_path", slidePath);

    const loadResponse = await fetch("/tileserver/slide", {
        method: "PUT",
        body: formData,
    });

    if (!loadResponse.ok) {
        throw new Error(`Failed to load slide: ${slidePath}`);
    }

    const metadataResponse = await fetch("/tileserver/slide");

    if (!metadataResponse.ok) {
        throw new Error("Failed to retrieve slide metadata.");
    }

    return metadataResponse.json();
}

// Get files from a directory configured when TileServer was launched.
async function getConfiguredFiles(kind) {
    const response = await fetch(`/tileserver/files/${kind}`);

    if (!response.ok) {
        throw new Error(`Failed to get configured ${kind} files.`);
    }

    return response.json();
}

async function clearOverlays() {
    const response = await fetch("/tileserver/clear_overlays", {
        method: "PUT",
    });

    if (!response.ok) {
        throw new Error("Failed to clear overlays.");
    }
}

async function removeSlide() {
    const response = await fetch("/tileserver/slide", {
        method: "DELETE",
    });

    if (!response.ok) {
        throw new Error("Failed to remove the current slide.");
    }
}

async function loadOverlay(overlayPath, layerName) {
    const formData = new FormData();
    formData.append("overlay_path", overlayPath);
    formData.append("layer_name", layerName);

    const response = await fetch("/tileserver/overlay", {
        method: "PUT",
        body: formData,
    });

    if (!response.ok) {
        throw new Error(`Failed to load overlay: ${overlayPath}`);
    }

    return response.json();
}

async function removeOverlay(layerName) {
    const response = await fetch(
        `/tileserver/overlay/${encodeURIComponent(layerName)}`,
        {
            method: "DELETE",
        },
    );

    if (!response.ok) {
        throw new Error(`Failed to remove overlay: ${layerName}`);
    }
}

async function setAnnotationColors(colorMap) {
    const formData = new FormData();
    formData.append(
        "cmap",
        JSON.stringify({
            keys: Object.keys(colorMap),
            values: Object.values(colorMap),
        }),
    );

    const response = await fetch("/tileserver/cmap", {
        method: "PUT",
        body: formData,
    });

    if (!response.ok) {
        throw new Error("Failed to update annotation colours.");
    }
}

export {
    clearOverlays,
    createSession,
    getConfiguredFiles,
    loadOverlay,
    loadSlide,
    removeOverlay,
    removeSlide,
    setAnnotationColors,
};
