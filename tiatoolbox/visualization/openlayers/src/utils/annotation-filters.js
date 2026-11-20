function getAnnotationFilter(
    annotationTypes,
    annotationTypeVisibility,
) {
    const visibleTypes =
        annotationTypes.filter(
            (annotationType) =>
                annotationTypeVisibility.get(
                    annotationType,
                ) ?? true,
        );

    if (
        visibleTypes.length ===
        annotationTypes.length
    ) {
        return null;
    }

    if (visibleTypes.length === 0) {
        const annotationType =
            JSON.stringify(
                annotationTypes[0],
            );

        return (
            `(props["type"]==${annotationType}) & ` +
            `(props["type"]!=${annotationType})`
        );
    }

    return visibleTypes
        .map(
            (annotationType) =>
                `(props["type"]==${JSON.stringify(annotationType)})`,
        )
        .join(" | ");
}

export {
    getAnnotationFilter,
};
