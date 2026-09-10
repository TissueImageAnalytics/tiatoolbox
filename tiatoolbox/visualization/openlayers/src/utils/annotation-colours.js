const annotationColourPalette = [
    [0.9, 0.05, 0.05, 1],       // red
    [0.05, 0.75, 0.2, 1],       // green
    [0, 0.8, 0.9, 1],           // cyan
    [0.95, 0.15, 0.65, 1],      // magenta
    [1, 0.85, 0.05, 1],         // yellow
    [0, 0.55, 0.4, 1],          // teal
    [1, 0.4, 0.05, 1],          // orange
    [0.5, 0.9, 0.05, 1],        // lime
    [1, 0.45, 0.65, 1],         // pink
    [0.15, 0.65, 1, 1],         // sky blue
    [0.65, 0.25, 0.85, 1],      // violet
    [0.1, 0.9, 0.55, 1],        // mint
    [0.9, 0.3, 0.25, 1],        // coral
    [0.65, 0.8, 0.05, 1],       // yellow-green
    [0.85, 0.1, 0.4, 1],        // raspberry
    [0.05, 0.6, 0.65, 1],       // dark cyan
    [0.95, 0.65, 0.1, 1],       // amber
    [0.4, 0.75, 0.35, 1],       // medium green
    [0.65, 0.45, 1, 1],         // light violet
    [0.2, 0.75, 0.75, 1],       // turquoise
];

function getAnnotationTypeKey(annotationType) {
    return `${typeof annotationType}:${String(annotationType)}`;
}

function hashAnnotationType(annotationType) {
    const key = getAnnotationTypeKey(annotationType);

    let hash = 2166136261;

    for (let index = 0; index < key.length; index += 1) {
        hash ^= key.charCodeAt(index);
        hash = Math.imul(
            hash,
            16777619,
        );
    }

    return hash >>> 0;
}

function getAnnotationColour(annotationType) {
    let paletteIndex;

    if (
        typeof annotationType === "number" &&
        Number.isInteger(annotationType) &&
        annotationType >= 0
    ) {
        paletteIndex =
            annotationType %
            annotationColourPalette.length;
    } else {
        paletteIndex =
            hashAnnotationType(annotationType) %
            annotationColourPalette.length;
    }

    return [
        ...annotationColourPalette[
            paletteIndex
        ],
    ];
}

function assignAnnotationColours(
    colourMap,
    annotationTypes,
) {
    for (const annotationType of new Set(
        annotationTypes,
    )) {
        if (colourMap.has(annotationType)) {
            continue;
        }

        colourMap.set(
            annotationType,
            getAnnotationColour(
                annotationType,
            ),
        );
    }

    return colourMap;
}

export {
    assignAnnotationColours,
};
