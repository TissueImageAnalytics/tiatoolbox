async function assignAnnotationColours(
    colourMap,
    annotationTypes,
    getColours,
    configuredColours = {},
) {
    const newTypes = [
        ...new Set(annotationTypes),
    ].filter(
        (annotationType) =>
            !colourMap.has(annotationType),
    );

    if (newTypes.length === 0) {
        return colourMap;
    }

    const generatedTypes =
        newTypes.filter(
            (annotationType) =>
                !Object.hasOwn(
                    configuredColours,
                    String(annotationType),
                ),
        );

    const generatedColours =
        generatedTypes.length === 0
            ? new Map()
            : await getColours(generatedTypes);

    for (const annotationType of newTypes) {
        const configuredColour =
            configuredColours[
                String(annotationType)
            ];

        colourMap.set(
            annotationType,
            configuredColour === undefined
                ? generatedColours.get(
                    annotationType,
                )
                : [
                    configuredColour[0] / 255,
                    configuredColour[1] / 255,
                    configuredColour[2] / 255,
                    1,
                ],
        );
    }

    return colourMap;
}

function isPlainObject(value) {
    return (
        value !== null &&
        typeof value === "object" &&
        !Array.isArray(value)
    );
}


function validateColourDict(
    colourDict,
    label,
) {
    if (!isPlainObject(colourDict)) {
        throw new Error(
            `${label} must be an object.`,
        );
    }

    for (const [
        annotationType,
        colour,
    ] of Object.entries(colourDict)) {
        if (
            !Array.isArray(colour) ||
            (
                colour.length !== 3 &&
                colour.length !== 4
            ) ||
            !colour.every(
                (channel) =>
                    typeof channel === "number" &&
                    Number.isFinite(channel) &&
                    channel >= 0 &&
                    channel <= 255,
            )
        ) {
            throw new Error(
                `Invalid annotation colour for ${label}.${annotationType}.`,
            );
        }
    }
}


function parseAnnotationColourConfig(config) {
    if (!isPlainObject(config)) {
        throw new Error(
            "Annotation colour config must be an object.",
        );
    }

    const hasColorDict =
        Object.hasOwn(
            config,
            "color_dict",
        );

    const hasLayerColorDicts =
        Object.hasOwn(
            config,
            "layer_color_dicts",
        );

    if (
        !hasColorDict &&
        !hasLayerColorDicts
    ) {
        throw new Error(
            "Annotation colour config must contain color_dict or layer_color_dicts.",
        );
    }

    const colorDict =
        hasColorDict
            ? config.color_dict
            : {};

    validateColourDict(
        colorDict,
        "color_dict",
    );

    const layerColorDicts = {};

    if (hasLayerColorDicts) {
        if (
            !isPlainObject(
                config.layer_color_dicts,
            )
        ) {
            throw new Error(
                "layer_color_dicts must be an object.",
            );
        }

        for (const [
            layerName,
            colourDict,
        ] of Object.entries(
            config.layer_color_dicts,
        )) {
            validateColourDict(
                colourDict,
                `layer_color_dicts.${layerName}`,
            );

            layerColorDicts[
                layerName
            ] = colourDict;
        }
    }

    return {
        colorDict,
        layerColorDicts,
    };
}


function mergeAnnotationColourConfig(
    colourMap,
    annotationTypes,
    colorDict,
    layerColorDict = {},
) {
    const updatedColours =
        new Map(colourMap);

    for (
        const annotationType of
        new Set(annotationTypes)
    ) {
        const key =
            String(annotationType);

        const configuredColour =
            Object.hasOwn(
                layerColorDict,
                key,
            )
                ? layerColorDict[key]
                : colorDict[key];

        if (configuredColour === undefined) {
            continue;
        }

        const currentColour =
            updatedColours.get(
                annotationType,
            );

        const opacity =
            currentColour?.[3] ?? 1;

        updatedColours.set(
            annotationType,
            [
                configuredColour[0] / 255,
                configuredColour[1] / 255,
                configuredColour[2] / 255,
                opacity,
            ],
        );
    }

    return updatedColours;
}

function createAnnotationColourConfig(
    colourMap,
    annotationTypes,
) {
    const colorDict = {};

    for (const annotationType of new Set(annotationTypes)) {
        const colour =
            colourMap.get(annotationType);

        if (colour === undefined) {
            continue;
        }

        colorDict[String(annotationType)] = [
            Math.round(colour[0] * 255),
            Math.round(colour[1] * 255),
            Math.round(colour[2] * 255),
            255,
        ];
    }

    return {
        color_dict: colorDict,
    };
}

export {
    assignAnnotationColours,
    createAnnotationColourConfig,
    mergeAnnotationColourConfig,
    parseAnnotationColourConfig,
};
