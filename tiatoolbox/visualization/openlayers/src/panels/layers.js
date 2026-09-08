import { getFileStem } from "../utils/paths.js";

// Manage layer visibility, opacity, ordering, removal and panel state.
function createLayersPanelController({
    panel,
    toggle,
    list,
    getSlideLayer,
    getCurrentSlidePath,
    getOverlayLayers,
    onRemoveLayer,
    onOpen,
}) {
    function setOpen(open) {
        if (open) {
            onOpen();
        }

        panel.classList.toggle("hidden", !open);
        toggle.classList.toggle("active", open);
    }

    function getEntries() {
        const entries = [];
        const slideLayer = getSlideLayer();

        if (slideLayer.getSource() !== null) {
            entries.push({
                id: "slide",
                name: getFileStem(
                    getCurrentSlidePath() ?? "slide",
                ),
                layer: slideLayer,
            });
        }

        const overlayEntries = Object.entries(
            getOverlayLayers(),
        )
            .map(([layerName, layer]) => ({
                id: layerName,
                name: layerName,
                layer,
            }))
            .sort(
                (a, b) =>
                    (a.layer.getZIndex() ?? 0) -
                    (b.layer.getZIndex() ?? 0),
            );

        entries.push(...overlayEntries);

        return entries;
    }

    function moveLayer(layerId, direction) {
        if (layerId === "slide") {
            return;
        }

        const entries = getEntries().filter(
            (entry) => entry.id !== "slide",
        );

        const index = entries.findIndex(
            (entry) => entry.id === layerId,
        );

        if (index === -1) {
            return;
        }

        const targetIndex =
            direction === "up"
                ? index - 1
                : index + 1;

        if (
            targetIndex < 0 ||
            targetIndex >= entries.length
        ) {
            return;
        }

        const currentLayer = entries[index].layer;
        const targetLayer =
            entries[targetIndex].layer;

        const currentZIndex =
            currentLayer.getZIndex() ?? 0;

        const targetZIndex =
            targetLayer.getZIndex() ?? 0;

        currentLayer.setZIndex(targetZIndex);
        targetLayer.setZIndex(currentZIndex);

        render();
    }

    function render() {
        list.replaceChildren();

        const entries = getEntries();

        const overlayEntries = entries.filter(
            (entry) => entry.id !== "slide",
        );

        if (entries.length === 0) {
            const empty =
                document.createElement("div");

            empty.className = "layer-editor-empty";
            empty.textContent = "No layers loaded";

            list.appendChild(empty);

            return;
        }

        entries.forEach(
            ({
                id: layerId,
                name: layerName,
                layer,
            }) => {
                const item =
                    document.createElement("div");

                item.className = "layer-editor-item";

                const header =
                    document.createElement("div");

                header.className =
                    "layer-editor-item-header";

                const visibility =
                    document.createElement("input");

                visibility.className =
                    "layer-editor-visibility";

                visibility.type = "checkbox";
                visibility.checked =
                    layer.getVisible();

                visibility.title =
                    `Toggle ${layerName}`;

                visibility.addEventListener(
                    "change",
                    () => {
                        layer.setVisible(
                            visibility.checked,
                        );
                    },
                );

                const name =
                    document.createElement("span");

                name.className =
                    "layer-editor-name";

                name.textContent = layerName;
                name.title = layerName;

                header.append(
                    visibility,
                    name,
                );

                if (layerId !== "slide") {
                    const overlayIndex =
                        overlayEntries.findIndex(
                            (entry) =>
                                entry.id === layerId,
                        );

                    const order =
                        document.createElement("div");

                    order.className =
                        "layer-editor-order";

                    const moveUp =
                        document.createElement("button");

                    moveUp.type = "button";
                    moveUp.title = "Move layer up";

                    moveUp.innerHTML =
                        '<i class="fas fa-chevron-up"></i>';

                    moveUp.disabled =
                        overlayIndex === 0;

                    moveUp.addEventListener(
                        "click",
                        () => {
                            moveLayer(
                                layerId,
                                "up",
                            );
                        },
                    );

                    const moveDown =
                        document.createElement("button");

                    moveDown.type = "button";
                    moveDown.title =
                        "Move layer down";

                    moveDown.innerHTML =
                        '<i class="fas fa-chevron-down"></i>';

                    moveDown.disabled =
                        overlayIndex ===
                        overlayEntries.length - 1;

                    moveDown.addEventListener(
                        "click",
                        () => {
                            moveLayer(
                                layerId,
                                "down",
                            );
                        },
                    );

                    const remove =
                        document.createElement("button");

                    remove.type = "button";
                    remove.title =
                        `Remove ${layerName}`;

                    remove.innerHTML =
                        '<i class="fas fa-times"></i>';

                    remove.addEventListener(
                        "click",
                        () => {
                            onRemoveLayer(layerId).catch(
                                (error) => {
                                    console.error(error);
                                },
                            );
                        },
                    );

                    order.append(
                        moveUp,
                        moveDown,
                        remove,
                    );

                    header.appendChild(order);
                }

                const opacityRow =
                    document.createElement("div");

                opacityRow.className =
                    "layer-editor-opacity";

                const slider =
                    document.createElement("input");

                slider.className =
                    "layer-editor-slider";

                slider.type = "range";
                slider.min = "0";
                slider.max = "1";
                slider.step = "0.05";

                slider.value =
                    layer.getOpacity().toString();

                const value =
                    document.createElement("span");

                value.className =
                    "layer-editor-value";

                value.textContent =
                    `${Math.round(
                        layer.getOpacity() * 100,
                    )}%`;

                slider.addEventListener(
                    "input",
                    () => {
                        const opacity =
                            Number(slider.value);

                        layer.setOpacity(opacity);

                        value.textContent =
                            `${Math.round(
                                opacity * 100,
                            )}%`;
                    },
                );

                opacityRow.append(
                    slider,
                    value,
                );

                item.append(
                    header,
                    opacityRow,
                );

                list.appendChild(item);
            },
        );
    }

    toggle.addEventListener("click", () => {
        const open =
            panel.classList.contains("hidden");

        setOpen(open);
    });

    return {
        render,
        setOpen,
    };
}

export { createLayersPanelController };
