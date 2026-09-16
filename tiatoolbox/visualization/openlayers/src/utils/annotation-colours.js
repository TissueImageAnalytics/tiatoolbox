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
};
