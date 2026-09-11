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

export {
    assignAnnotationColours,
};
