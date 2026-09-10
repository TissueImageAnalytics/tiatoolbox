async function assignAnnotationColours(
    colourMap,
    annotationTypes,
    getColours,
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

    const generatedColours =
        await getColours(newTypes);

    for (const annotationType of newTypes) {
        colourMap.set(
            annotationType,
            generatedColours.get(
                annotationType,
            ),
        );
    }

    return colourMap;
}

export {
    assignAnnotationColours,
};
