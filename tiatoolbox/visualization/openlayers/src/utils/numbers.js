function getFiniteNumberRange(values) {
    if (values.length === 0) {
        return null;
    }

    let minimum = Infinity;
    let maximum = -Infinity;

    for (const value of values) {
        if (
            typeof value !== "number" ||
            !Number.isFinite(value)
        ) {
            return null;
        }

        minimum = Math.min(
            minimum,
            value,
        );

        maximum = Math.max(
            maximum,
            value,
        );
    }

    return [
        minimum,
        maximum,
    ];
}

export {
    getFiniteNumberRange,
};
