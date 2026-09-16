// Manage annotation type controls without owning application state.
function colourToHex(colour) {
    if (typeof colour === "string") {
        return colour;
    }

    return `#${colour
        .slice(0, 3)
        .map((channel) =>
            Math.round(channel * 255)
                .toString(16)
                .padStart(2, "0"))
        .join("")}`;
}

function createAnnotationsPanelController({
    panel,
    toggle,
    list,
    selectAllButton,
    deselectAllButton,
    exportButton,
    colourBySelect,
    secondaryTypeField,
    secondaryTypeSelect,
    propertyField,
    propertySelect,
    propertyLegend,
    propertyLegendCaption,
    propertyMin,
    propertyMax,
    getAnnotationGroups,
    getAnnotationTypes,
    getAnnotationColour,
    getDisplayMode,
    getAnnotationProperties,
    getAnnotationProperty,
    getSecondaryType,
    getPropertyRange,
    isAnnotationTypeVisible,
    getAnnotationOpacity,
    onColourChange,
    onVisibilityChange,
    onOpacityChange,
    onDisplayModeChange,
    onPropertyChange,
    onSecondaryTypeChange,
    onSetAllVisibility,
    onExport,
    onOpen,
}) {
    function setOpen(open) {
        if (open) {
            onOpen();
            render();
        }

        panel.classList.toggle("hidden", !open);
        toggle.classList.toggle("active", open);
    }

    function runAction(action) {
        action()
            .then(() => {
                render();
            })
            .catch((error) => {
                console.error(error);
                render();
            });
    }

    function updatePropertyOptions(
        properties,
    ) {
        const currentProperties =
            [...propertySelect.options].map(
                (option) => option.value,
            );

        if (
            currentProperties.length ===
                properties.length &&
            currentProperties.every(
                (property, index) =>
                    property ===
                    properties[index],
            )
        ) {
            return;
        }

        const options =
            properties.map(
                (property) => {
                    const option =
                        document.createElement(
                            "option",
                        );

                    option.value =
                        property;

                    option.textContent =
                        property;

                    return option;
                },
            );

        propertySelect.replaceChildren(
            ...options,
        );
    }

    function updateSecondaryTypeOptions(
        annotationTypes,
    ) {
        const values =
            annotationTypes.map(
                (annotationType) =>
                    JSON.stringify(
                        annotationType,
                    ),
            );

        const currentValues =
            [...secondaryTypeSelect.options].map(
                (option) => option.value,
            );

        if (
            currentValues.length ===
                values.length &&
            currentValues.every(
                (value, index) =>
                    value ===
                    values[index],
            )
        ) {
            return;
        }

        const options =
            annotationTypes.map(
                (annotationType) => {
                    const option =
                        document.createElement(
                            "option",
                        );

                    option.value =
                        JSON.stringify(
                            annotationType,
                        );

                    option.textContent =
                        String(
                            annotationType,
                        );

                    return option;
                },
            );

        secondaryTypeSelect.replaceChildren(
            ...options,
        );
    }

    function render() {
        list.replaceChildren();

        const annotationGroups =
            getAnnotationGroups();

        const displayMode =
            getDisplayMode();

        const properties =
            getAnnotationProperties();

        const annotationTypes =
            getAnnotationTypes();

        const selectedSecondaryType =
            getSecondaryType();

        const hasAnnotations =
            annotationGroups.length > 0;

        colourBySelect.value =
            displayMode;

        colourBySelect.disabled =
            !hasAnnotations;

        colourBySelect
            .closest(
                ".annotations-panel-display-field",
            )
            ?.classList.toggle(
                "disabled",
                !hasAnnotations,
            );

        const propertyModeOption =
            colourBySelect.querySelector(
                'option[value="property"]',
            );

        const secondaryModeOption =
            colourBySelect.querySelector(
                'option[value="secondary"]',
            );

        propertyModeOption.disabled =
            !hasAnnotations ||
            properties.length === 0;

        secondaryModeOption.disabled =
            !hasAnnotations ||
            annotationTypes.length === 0 ||
            properties.length === 0;

        secondaryTypeField.hidden =
            displayMode !== "secondary";

        propertyField.hidden =
            displayMode === "type";

        updatePropertyOptions(
            properties,
        );

        updateSecondaryTypeOptions(
            annotationTypes,
        );

        const selectedProperty =
            getAnnotationProperty();

        if (
            selectedProperty !== null &&
            properties.includes(
                selectedProperty,
            )
        ) {
            propertySelect.value =
                selectedProperty;
        }

        if (
            selectedSecondaryType !== null &&
            annotationTypes.some(
                (annotationType) =>
                    Object.is(
                        annotationType,
                        selectedSecondaryType,
                    ),
            )
        ) {
            secondaryTypeSelect.value =
                JSON.stringify(
                    selectedSecondaryType,
                );
        }

        secondaryTypeSelect.disabled =
            annotationTypes.length === 0;

        const propertyRange =
            getPropertyRange();

        const showPropertyLegend =
            (
                displayMode === "property" ||
                displayMode === "secondary"
            ) &&
            propertyRange !== null;

        propertyLegend.hidden =
            !showPropertyLegend;

        if (showPropertyLegend) {
            const [
                minimum,
                maximum,
            ] = propertyRange;

            propertyLegendCaption.textContent =
                displayMode === "secondary"
                    ? `${String(
                        selectedSecondaryType,
                    )} · ${selectedProperty} values · low → high`
                    : `${selectedProperty} values · low → high`;

            propertyMin.textContent =
                Number(
                    minimum.toPrecision(4),
                ).toString();

            propertyMax.textContent =
                Number(
                    maximum.toPrecision(4),
                ).toString();
        }

        propertySelect.disabled =
            properties.length === 0;

        selectAllButton.disabled =
            !hasAnnotations;

        deselectAllButton.disabled =
            !hasAnnotations;

        exportButton.disabled =
            !hasAnnotations ||
            displayMode === "property";

        if (annotationGroups.length === 0) {
            const empty =
                document.createElement("div");

            empty.className =
                "annotations-panel-empty";

            empty.textContent =
                "No annotations loaded";

            list.appendChild(empty);

            return;
        }

        for (const {
            layerName,
            annotationTypes,
        } of annotationGroups) {
            const group =
                document.createElement("section");

            group.className =
                "annotations-panel-group";

            const groupTitle =
                document.createElement("div");

            groupTitle.className =
                "annotations-panel-group-title";

            groupTitle.textContent =
                layerName;

            groupTitle.title =
                layerName;

            group.appendChild(
                groupTitle,
            );

            for (const annotationType of annotationTypes) {
                const annotationName =
                    String(annotationType);

                const secondaryTypeSelected =
                    displayMode === "secondary" &&
                    selectedSecondaryType !== null &&
                    Object.is(
                        annotationType,
                        selectedSecondaryType,
                    );

                const item =
                    document.createElement("div");

                item.className =
                    "annotations-panel-item";

                const header =
                    document.createElement("div");

                header.className =
                    "annotations-panel-item-header";

                const visibility =
                    document.createElement("input");

                visibility.type = "checkbox";
                visibility.className =
                    "annotations-panel-visibility";

                visibility.checked =
                    isAnnotationTypeVisible(
                        annotationType,
                    );

                visibility.title =
                    `Toggle ${annotationName}`;

                visibility.addEventListener(
                    "change",
                    () => {
                        runAction(() =>
                            onVisibilityChange(
                                annotationType,
                                visibility.checked,
                            ));
                    },
                );

                const colour =
                    document.createElement("input");

                colour.type = "color";
                colour.className =
                    "annotations-panel-colour";

                colour.value =
                    colourToHex(
                        getAnnotationColour(
                            annotationType,
                        ),
                    );

                colour.title =
                    `Change ${annotationName} colour`;

                colour.disabled =
                    displayMode === "property" ||
                    secondaryTypeSelected;

                colour.addEventListener(
                    "change",
                    () => {
                        runAction(() =>
                            onColourChange(
                                annotationType,
                                colour.value,
                            ));
                    },
                );

                const name =
                    document.createElement("span");

                name.className =
                    "annotations-panel-name";

                name.textContent =
                    annotationName;

                name.title =
                    annotationName;

                header.append(
                    visibility,
                    colour,
                    name,
                );

                const opacityRow =
                    document.createElement("div");

                opacityRow.className =
                    "annotations-panel-opacity";

                const opacityLabel =
                    document.createElement("span");

                opacityLabel.className =
                    "annotations-panel-opacity-label";

                opacityLabel.textContent =
                    "Fill opacity";

                const slider =
                    document.createElement("input");

                slider.type = "range";
                slider.className =
                    "annotations-panel-slider";

                slider.min = "0";
                slider.max = "1";
                slider.step = "0.05";

                slider.value =
                    getAnnotationOpacity(
                        annotationType,
                    ).toString();

                slider.disabled =
                    displayMode === "property" ||
                    secondaryTypeSelected;

                const value =
                    document.createElement("span");

                value.className =
                    "annotations-panel-value";

                value.textContent =
                    `${Math.round(
                        Number(slider.value) * 100,
                    )}%`;

                slider.addEventListener(
                    "input",
                    () => {
                        value.textContent =
                            `${Math.round(
                                Number(slider.value) * 100,
                            )}%`;
                    },
                );

                slider.addEventListener(
                    "change",
                    () => {
                        runAction(() =>
                            onOpacityChange(
                                annotationType,
                                Number(slider.value),
                            ));
                    },
                );

                opacityRow.append(
                    opacityLabel,
                    slider,
                    value,
                );

                item.append(
                    header,
                    opacityRow,
                );

                group.appendChild(item);
            }

            list.appendChild(group);
        }
    }

    colourBySelect.addEventListener(
        "change",
        () => {
            runAction(() =>
                onDisplayModeChange(
                    colourBySelect.value,
                ));
        },
    );

    propertySelect.addEventListener(
        "change",
        () => {
            runAction(() =>
                onPropertyChange(
                    propertySelect.value,
                ));
        },
    );

    secondaryTypeSelect.addEventListener(
        "change",
        () => {
            runAction(() =>
                onSecondaryTypeChange(
                    JSON.parse(
                        secondaryTypeSelect.value,
                    ),
                ));
        },
    );

    selectAllButton.addEventListener(
        "click",
        () => {
            runAction(() =>
                onSetAllVisibility(true));
        },
    );

    deselectAllButton.addEventListener(
        "click",
        () => {
            runAction(() =>
                onSetAllVisibility(false));
        },
    );

    exportButton.addEventListener(
        "click",
        () => {
            onExport();
        },
    );

    toggle.addEventListener("click", () => {
        setOpen(
            panel.classList.contains("hidden"),
        );
    });

    return {
        render,
        setOpen,
    };
}

export {
    createAnnotationsPanelController,
};
