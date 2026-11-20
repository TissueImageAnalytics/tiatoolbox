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
    importButton,
    importInput,
    exportButton,
    colourBySelect,
    paletteField,
    paletteSelect,
    colourMapField,
    colourMapSelect,
    secondaryTypeField,
    secondaryTypeSelect,
    propertyField,
    propertySelect,
    propertyLegend,
    propertyLegendCaption,
    propertyMin,
    propertyMax,
    linkOpacityInput,
    getAnnotationGroups,
    getAnnotationTypes,
    getAnnotationColour,
    getDisplayMode,
    getPalette,
    getColourMap,
    getAnnotationProperties,
    getAnnotationProperty,
    getSecondaryType,
    getPropertyRange,
    isAnnotationTypeVisible,
    getAnnotationOpacity,
    getOpacityLinked,
    onColourChange,
    onVisibilityChange,
    onOpacityChange,
    onOpacityLinkChange,
    onPaletteChange,
    onColourMapChange,
    onDisplayModeChange,
    onPropertyChange,
    onSecondaryTypeChange,
    onSetAllVisibility,
    onImport,
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
        annotationGroups,
    ) {
        const selections =
            annotationGroups.flatMap(
                ({
                    layerName,
                    annotationTypes,
                }) =>
                    annotationTypes.map(
                        (annotationType) => ({
                            layerName,
                            annotationType,
                        }),
                    ),
            );

        const values =
            selections.map(
                ({
                    layerName,
                    annotationType,
                }) =>
                    JSON.stringify([
                        layerName,
                        annotationType,
                    ]),
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
                    value === values[index],
            )
        ) {
            return;
        }

        const options =
            selections.map(
                ({
                    layerName,
                    annotationType,
                }) => {
                    const option =
                        document.createElement(
                            "option",
                        );

                    option.value =
                        JSON.stringify([
                            layerName,
                            annotationType,
                        ]);

                    option.textContent =
                        `${layerName} · ${String(
                            annotationType,
                        )}`;

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

        const palette =
            getPalette();

        const colourMap =
            getColourMap();

        const properties =
            getAnnotationProperties();

        const annotationTypes =
            getAnnotationTypes();

        const selectedSecondarySelection =
            getSecondaryType();

        const hasAnnotations =
            annotationGroups.length > 0;

        linkOpacityInput.checked =
            getOpacityLinked();

        linkOpacityInput.disabled =
            !hasAnnotations;

        linkOpacityInput
            .closest(
                ".annotations-panel-link-opacity",
            )
            ?.classList.toggle(
                "disabled",
                !hasAnnotations,
            );

        paletteField.hidden =
            displayMode !== "type";

        colourMapField.hidden =
            displayMode === "type";

        paletteSelect.value =
            palette;

        colourMapSelect.value =
            colourMap;

        paletteSelect.disabled =
            !hasAnnotations;

        colourMapSelect.disabled =
            !hasAnnotations ||
            properties.length === 0;

        paletteField.classList.toggle(
            "disabled",
            !hasAnnotations,
        );

        colourMapField.classList.toggle(
            "disabled",
            !hasAnnotations ||
            properties.length === 0,
        );

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
            annotationGroups,
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
            selectedSecondarySelection !== null
        ) {
            const selectedValue =
                JSON.stringify([
                    selectedSecondarySelection.layerName,
                    selectedSecondarySelection.annotationType,
                ]);

            if (
                [...secondaryTypeSelect.options].some(
                    (option) =>
                        option.value ===
                        selectedValue,
                )
            ) {
                secondaryTypeSelect.value =
                    selectedValue;
            }
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

        propertyLegend.dataset.colourMap =
            colourMap;

        if (showPropertyLegend) {
            const [
                minimum,
                maximum,
            ] = propertyRange;

            if (
                displayMode === "secondary" &&
                selectedSecondarySelection !== null
            ) {
                propertyLegendCaption.textContent =
                    `${selectedSecondarySelection.layerName} · ${String(
                        selectedSecondarySelection.annotationType,
                    )} · ${selectedProperty} values · low → high`;
            } else {
                propertyLegendCaption.textContent =
                    `${selectedProperty} values · low → high`;
            }

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
            !hasAnnotations ||
            displayMode === "secondary";

        deselectAllButton.disabled =
            !hasAnnotations ||
            displayMode === "secondary";

        importButton.disabled =
            !hasAnnotations ||
            displayMode !== "type";

        exportButton.disabled =
            !hasAnnotations ||
            displayMode !== "type";

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
                    selectedSecondarySelection !== null &&
                    layerName ===
                        selectedSecondarySelection.layerName &&
                    Object.is(
                        annotationType,
                        selectedSecondarySelection.annotationType,
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
                    displayMode === "secondary"
                        ? secondaryTypeSelected
                        : isAnnotationTypeVisible(
                            layerName,
                            annotationType,
                        );

                visibility.disabled =
                    displayMode === "secondary";

                visibility.title =
                    `Toggle ${annotationName}`;

                visibility.addEventListener(
                    "change",
                    () => {
                        runAction(() =>
                            onVisibilityChange(
                                layerName,
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
                            layerName,
                            annotationType,
                        )
                    );

                colour.title =
                    `Change ${annotationName} colour`;

                colour.disabled =
                    displayMode === "property" ||
                    displayMode === "secondary";

                colour.addEventListener(
                    "change",
                    () => {
                        runAction(() =>
                            onColourChange(
                                layerName,
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
                        layerName,
                        annotationType,
                    ).toString();

                slider.disabled =
                    displayMode === "secondary" &&
                    !secondaryTypeSelected;

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
                        const percentage =
                            `${Math.round(
                                Number(slider.value) * 100,
                            )}%`;

                        value.textContent =
                            percentage;

                        if (linkOpacityInput.checked) {
                            for (
                                const linkedSlider of
                                list.querySelectorAll(
                                    ".annotations-panel-slider",
                                )
                            ) {
                                linkedSlider.value =
                                    slider.value;

                                linkedSlider
                                    .closest(
                                        ".annotations-panel-opacity",
                                    )
                                    ?.querySelector(
                                        ".annotations-panel-value",
                                    )
                                    ?.replaceChildren(
                                        percentage,
                                    );
                            }
                        }
                    },
                );

                slider.addEventListener(
                    "change",
                    () => {
                        runAction(() =>
                            onOpacityChange(
                                layerName,
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

    linkOpacityInput.addEventListener(
        "change",
        () => {
            runAction(() =>
                onOpacityLinkChange(
                    linkOpacityInput.checked,
                ));
        },
    );

    importButton.addEventListener(
        "click",
        () => {
            importInput.click();
        },
    );

    importInput.addEventListener(
        "change",
        () => {
            const file =
                importInput.files?.[0];

            importInput.value = "";

            if (file === undefined) {
                return;
            }

            runAction(() =>
                onImport(file));
        },
    );

    paletteSelect.addEventListener(
        "change",
        () => {
            runAction(() =>
                onPaletteChange(
                    paletteSelect.value,
                ));
        },
    );

    colourMapSelect.addEventListener(
        "change",
        () => {
            runAction(() =>
                onColourMapChange(
                    colourMapSelect.value,
                ));
        },
    );

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
            const [
                layerName,
                annotationType,
            ] = JSON.parse(
                secondaryTypeSelect.value,
            );

            runAction(() =>
                onSecondaryTypeChange(
                    layerName,
                    annotationType,
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
