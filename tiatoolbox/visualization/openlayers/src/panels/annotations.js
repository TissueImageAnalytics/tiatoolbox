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
    showAllButton,
    hideAllButton,
    exportButton,
    getAnnotationGroups,
    getAnnotationColour,
    isAnnotationTypeVisible,
    getAnnotationOpacity,
    onColourChange,
    onVisibilityChange,
    onOpacityChange,
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

    function render() {
        list.replaceChildren();

        const annotationGroups =
            getAnnotationGroups();

        const hasAnnotations =
            annotationGroups.length > 0;

        showAllButton.disabled =
            !hasAnnotations;

        hideAllButton.disabled =
            !hasAnnotations;

        exportButton.disabled =
            !hasAnnotations;

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

    showAllButton.addEventListener(
        "click",
        () => {
            runAction(() =>
                onSetAllVisibility(true));
        },
    );

    hideAllButton.addEventListener(
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
