// TileServer API helpers for dynamic slide and overlay management.

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

function getAnnotationRendererUrl(
    path,
    layerName,
) {
    if (layerName === null) {
        return path;
    }

    const params =
        new URLSearchParams({
            layer: layerName,
        });

    return `${path}?${params}`;
}

async function setAnnotationFilter(
    where,
    layerName = null,
) {
    const formData = new FormData();

    formData.append(
        "val",
        JSON.stringify(where),
    );

    const response = await fetch(
        getAnnotationRendererUrl(
            "/tileserver/renderer/where",
            layerName,
        ),
        {
            method: "PUT",
            body: formData,
        },
    );

    if (!response.ok) {
        throw new Error(
            "Failed to update annotation visibility.",
        );
    }
}

async function setAnnotationColors(
    colorMap,
    layerName = null,
) {
    const entries =
        colorMap instanceof Map
            ? [...colorMap.entries()]
            : Object.entries(colorMap);

    const formData = new FormData();

    formData.append(
        "cmap",
        JSON.stringify({
            keys: entries.map(
                ([key]) => key,
            ),
            values: entries.map(
                ([, value]) => value,
            ),
        }),
    );

    const response = await fetch(
        getAnnotationRendererUrl(
            "/tileserver/cmap",
            layerName,
        ),
        {
            method: "PUT",
            body: formData,
        },
    );

    if (!response.ok) {
        throw new Error("Failed to update annotation colours.");
    }
}

async function getAnnotationColors(annotationTypes) {
    const formData = new FormData();

    formData.append(
        "types",
        JSON.stringify(annotationTypes),
    );

    const response = await fetch(
        "/tileserver/annotation_colours",
        {
            method: "PUT",
            body: formData,
        },
    );

    if (!response.ok) {
        throw new Error(
            "Failed to generate annotation colours.",
        );
    }

    const colourMap =
        await response.json();

    return new Map(
        colourMap.keys.map(
            (key, index) => [
                key,
                colourMap.values[index],
            ],
        ),
    );
}

async function getAnnotationProperties(layerName) {
    const params =
        new URLSearchParams({
            layer: layerName,
        });

    const response = await fetch(
        `/tileserver/prop_names/all?${params}`,
    );

    if (!response.ok) {
        throw new Error(
            "Failed to get annotation properties.",
        );
    }

    return response.json();
}

async function getAnnotationPropertyValues(
    layerName,
    property,
) {
    const params =
        new URLSearchParams({
            layer: layerName,
        });

    const response = await fetch(
        `/tileserver/prop_values/${encodeURIComponent(property)}/all?${params}`,
    );

    if (!response.ok) {
        throw new Error(
            "Failed to get annotation property values.",
        );
    }

    return response.json();
}

async function getAnnotationAtPoint(
    layerName,
    x,
    y,
) {
    const params =
        new URLSearchParams({
            layer: layerName,
        });

    const response = await fetch(
        `/tileserver/tap_query/${x}/${y}?${params}`,
    );

    if (!response.ok) {
        throw new Error(
            "Failed to inspect annotation.",
        );
    }

    return response.json();
}

async function setAnnotationProperty(
    property,
    layerName = null,
) {
    const formData = new FormData();

    formData.append(
        "val",
        JSON.stringify(property),
    );

    const response = await fetch(
        getAnnotationRendererUrl(
            "/tileserver/renderer/score_prop",
            layerName,
        ),
        {
            method: "PUT",
            body: formData,
        },
    );

    if (!response.ok) {
        throw new Error(
            "Failed to update annotation property.",
        );
    }
}

async function setAnnotationMapper(
    mapper,
    layerName = null,
) {
    const formData = new FormData();

    formData.append(
        "cmap",
        JSON.stringify(mapper),
    );

    const response = await fetch(
        getAnnotationRendererUrl(
            "/tileserver/cmap",
            layerName,
        ),
        {
            method: "PUT",
            body: formData,
        },
    );

    if (!response.ok) {
        throw new Error(
            "Failed to update annotation colour map.",
        );
    }
}

async function setAnnotationPropertyRange(
    range,
    layerName = null,
) {
    const formData = new FormData();

    formData.append(
        "range",
        JSON.stringify(range),
    );

    const response = await fetch(
        getAnnotationRendererUrl(
            "/tileserver/prop_range",
            layerName,
        ),
        {
            method: "PUT",
            body: formData,
        },
    );

    if (!response.ok) {
        throw new Error(
            "Failed to update annotation property range.",
        );
    }
}

async function setAnnotationSecondaryMapper(
    annotationType,
    property,
    mapper,
    range,
    layerName = null,
) {
    const formData = new FormData();

    formData.append(
        "type_id",
        JSON.stringify(
            annotationType,
        ),
    );

    formData.append(
        "prop",
        property,
    );

    formData.append(
        "cmap",
        JSON.stringify(
            mapper,
        ),
    );

    formData.append(
        "range",
        JSON.stringify(
            range,
        ),
    );

    const response = await fetch(
        getAnnotationRendererUrl(
            "/tileserver/secondary_cmap",
            layerName,
        ),
        {
            method: "PUT",
            body: formData,
        },
    );

    if (!response.ok) {
        throw new Error(
            "Failed to update secondary annotation colour map.",
        );
    }
}

async function clearAnnotationSecondaryMapper(
    layerName = null,
) {
    const formData = new FormData();

    formData.append(
        "val",
        JSON.stringify(null),
    );

    const response = await fetch(
        getAnnotationRendererUrl(
            "/tileserver/renderer/secondary_cmap",
            layerName,
        ),
        {
            method: "PUT",
            body: formData,
        },
    );

    if (!response.ok) {
        throw new Error(
            "Failed to clear secondary annotation colour map.",
        );
    }
}

export {
    clearAnnotationSecondaryMapper,
    clearOverlays,
    createSession,
    getConfiguredFiles,
    loadOverlay,
    loadSlide,
    removeOverlay,
    removeSlide,
    getAnnotationColors,
    getAnnotationAtPoint,
    getAnnotationProperties,
    getAnnotationPropertyValues,
    setAnnotationColors,
    setAnnotationFilter,
    setAnnotationMapper,
    setAnnotationProperty,
    setAnnotationPropertyRange,
    setAnnotationSecondaryMapper,
};
