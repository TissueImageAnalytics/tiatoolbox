// Manage viewer loading state.
function createLoadingIndicatorController({
    map,
    element,
    onChange = () => {},
}) {
    let pendingOperations = 0;
    let mapLoading = true;

    function isLoading() {
        return pendingOperations > 0 || mapLoading;
    }

    function update() {
        const loading = isLoading();

        element.hidden = !loading;
        onChange(loading);
    }

    function begin() {
        pendingOperations += 1;
        update();
    }

    function end() {
        pendingOperations = Math.max(
            0,
            pendingOperations - 1,
        );

        if (pendingOperations === 0) {
            mapLoading = true;
            map.render();
        }

        update();
    }

    map.on("loadstart", () => {
        mapLoading = true;
        update();
    });

    map.on("rendercomplete", () => {
        mapLoading = false;
        update();
    });

    update();
    map.render();

    return {
        begin,
        end,
        isLoading,
    };
}

export { createLoadingIndicatorController };
