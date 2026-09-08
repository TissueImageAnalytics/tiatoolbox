import {
    beforeEach,
    describe,
    expect,
    it,
    vi,
} from "vitest";

import {
    createGridController,
} from "../../../tiatoolbox/visualization/openlayers/src/controls/grid.js";

function createCheckbox(checked = true) {
    const input = document.createElement("input");

    input.type = "checkbox";
    input.checked = checked;

    return input;
}

function createInput(value) {
    const input = document.createElement("input");

    input.type = "range";
    input.min = "0";
    input.max = "100";
    input.value = value;

    return input;
}

function createSelect(values, value) {
    const select = document.createElement("select");

    const optionValues = values.includes(value)
        ? values
        : [
            ...values,
            value,
        ];

    for (const optionValue of optionValues) {
        const option = document.createElement("option");

        option.value = optionValue;
        option.textContent = optionValue;

        select.append(option);
    }

    select.value = value;

    return select;
}

function setSelectValue(select, value) {
    if (
        ![
            ...select.options,
        ].some(
            (option) => option.value === value,
        )
    ) {
        const option = document.createElement("option");

        option.value = value;
        option.textContent = value;

        select.append(option);
    }

    select.value = value;
}

function dispatchChange(element) {
    element.dispatchEvent(
        new Event(
            "change",
            {
                bubbles: true,
            },
        ),
    );
}

function dispatchInput(element) {
    element.dispatchEvent(
        new Event(
            "input",
            {
                bubbles: true,
            },
        ),
    );
}

function createProjection(code = "EPSG:3857") {
    return {
        getCode: vi.fn(
            () => code,
        ),
    };
}

function getMethodOwner(object, methodName) {
    let current = object;

    while (current !== null) {
        if (
            Object.prototype.hasOwnProperty.call(
                current,
                methodName,
            )
        ) {
            return current;
        }

        current = Object.getPrototypeOf(current);
    }

    throw new Error(
        `Could not find ${methodName}.`,
    );
}

function stubGraticuleSetMap(graticule) {
    const assignedMaps = new WeakMap();

    const methodOwner = getMethodOwner(
        graticule,
        "setMap",
    );

    const setMap = vi.spyOn(
        methodOwner,
        "setMap",
    ).mockImplementation(
        function setMapMock(map) {
            assignedMaps.set(
                this,
                map,
            );
        },
    );

    return {
        setMap,

        getAssignedMap(control) {
            return assignedMaps.has(control)
                ? assignedMaps.get(control)
                : null;
        },
    };
}

function clickToggle(toggle) {
    const button =
        toggle.element.querySelector(
            "button",
        );

    expect(button).not.toBeNull();

    button.click();
}

function createHarness({
    theme = "dark",
    gridTheme = "default",
    opacity = "50",
    spacing = "default",
    labelsVisible = true,
    graticuleVisible = true,
    screenSpaceGraticuleVisible = true,
} = {}) {
    const projection =
        createProjection();

    const view = {
        getProjection: vi.fn(
            () => projection,
        ),

        calculateExtent: vi.fn(
            () => [
                100,
                200,
                900,
                1000,
            ],
        ),

        getResolution: vi.fn(
            () => 2,
        ),
    };

    const map = {
        addControl: vi.fn(),
        renderSync: vi.fn(),

        getView: vi.fn(
            () => view,
        ),

        getSize: vi.fn(
            () => [
                800,
                600,
            ],
        ),
    };

    const themeSelect =
        createSelect(
            [
                "dark",
                "light",
                "high-contrast",
            ],
            theme,
        );

    const gridThemeSelect =
        createSelect(
            [
                "default",
                "light",
                "dark",
                "light-contrast",
                "dark-contrast",
            ],
            gridTheme,
        );

    const gridOpacityInput =
        createInput(opacity);

    const gridOpacityValue =
        document.createElement(
            "span",
        );

    const gridSpacingSelect =
        createSelect(
            [
                "fine",
                "default",
                "coarse",
            ],
            spacing,
        );

    const gridLabelsVisibleInput =
        createCheckbox(
            labelsVisible,
        );

    const graticuleVisibleInput =
        createCheckbox(
            graticuleVisible,
        );

    const screenSpaceGraticuleVisibleInput =
        createCheckbox(
            screenSpaceGraticuleVisible,
        );

    const onGraticulesChange =
        vi.fn();

    document.body.append(
        themeSelect,
        gridThemeSelect,
        gridOpacityInput,
        gridOpacityValue,
        gridSpacingSelect,
        gridLabelsVisibleInput,
        graticuleVisibleInput,
        screenSpaceGraticuleVisibleInput,
    );

    const controller =
        createGridController({
            map,
            projection,
            themeSelect,
            gridThemeSelect,
            gridOpacityInput,
            gridOpacityValue,
            gridSpacingSelect,
            gridLabelsVisibleInput,
            graticuleVisibleInput,
            screenSpaceGraticuleVisibleInput,
            onGraticulesChange,
        });

    const [
        graticule,
        screenSpaceGraticule,
    ] = onGraticulesChange.mock.calls[0];

    return {
        map,
        view,
        projection,
        themeSelect,
        gridThemeSelect,
        gridOpacityInput,
        gridOpacityValue,
        gridSpacingSelect,
        gridLabelsVisibleInput,
        graticuleVisibleInput,
        screenSpaceGraticuleVisibleInput,
        onGraticulesChange,
        controller,
        graticule,
        screenSpaceGraticule,
        graticuleToggle:
            controller.graticuleToggle,
        screenSpaceGraticuleToggle:
            controller.screenSpaceGraticuleToggle,
    };
}

function getGridStyle(graticule) {
    return graticule.getStyle();
}

beforeEach(() => {
    document.body.replaceChildren();
});

describe("initialisation", () => {
    it("creates both graticules and adds both toggle controls", () => {
        const {
            map,
            graticule,
            screenSpaceGraticule,
            graticuleToggle,
            screenSpaceGraticuleToggle,
            onGraticulesChange,
        } = createHarness();

        expect(
            graticule,
        ).toBeDefined();

        expect(
            screenSpaceGraticule,
        ).toBeDefined();

        expect(
            map.addControl,
        ).toHaveBeenCalledTimes(2);

        expect(
            map.addControl,
        ).toHaveBeenNthCalledWith(
            1,
            graticuleToggle,
        );

        expect(
            map.addControl,
        ).toHaveBeenNthCalledWith(
            2,
            screenSpaceGraticuleToggle,
        );

        expect(
            onGraticulesChange,
        ).toHaveBeenCalledExactlyOnceWith(
            graticule,
            screenSpaceGraticule,
        );
    });

    it("applies the initial appearance and labels", () => {
        const {
            map,
            graticule,
            gridOpacityValue,
        } = createHarness();

        const style =
            getGridStyle(
                graticule,
            );

        expect(
            style
                .getStroke()
                .getColor(),
        ).toBe(
            "rgba(20, 20, 20, 0.5)",
        );

        expect(
            style.getText(),
        ).not.toBeNull();

        expect(
            gridOpacityValue.textContent,
        ).toBe("50%");

        expect(
            map.renderSync,
        ).toHaveBeenCalledTimes(2);
    });

    it("starts without labels when label visibility is disabled", () => {
        const {
            graticule,
        } = createHarness({
            labelsVisible: false,
        });

        expect(
            getGridStyle(
                graticule,
            ).getText(),
        ).toBeNull();
    });
});

describe("appearance", () => {
    it.each([
        [
            "light",
            "rgba(255, 255, 255, 0.5)",
        ],
        [
            "dark",
            "rgba(20, 20, 20, 0.5)",
        ],
        [
            "light-contrast",
            "rgba(0, 170, 200, 0.5)",
        ],
        [
            "dark-contrast",
            "rgba(145, 55, 0, 0.5)",
        ],
    ])(
        "uses the %s grid theme",
        (
            gridTheme,
            expectedColour,
        ) => {
            const {
                graticule,
            } = createHarness({
                gridTheme,
            });

            expect(
                getGridStyle(
                    graticule,
                )
                    .getStroke()
                    .getColor(),
            ).toBe(expectedColour);
        },
    );

    it.each([
        [
            "dark",
            "rgba(20, 20, 20, 0.5)",
        ],
        [
            "light",
            "rgba(255, 255, 255, 0.5)",
        ],
        [
            "high-contrast",
            "rgba(145, 55, 0, 0.5)",
        ],
    ])(
        "uses the %s interface theme when the grid theme is default",
        (
            theme,
            expectedColour,
        ) => {
            const {
                graticule,
            } = createHarness({
                theme,
                gridTheme: "default",
            });

            expect(
                getGridStyle(
                    graticule,
                )
                    .getStroke()
                    .getColor(),
            ).toBe(expectedColour);
        },
    );

    it("falls back to dark colours for an unknown grid theme", () => {
        const {
            graticule,
        } = createHarness({
            gridTheme: "unknown",
        });

        const style =
            getGridStyle(
                graticule,
            );

        expect(
            style
                .getStroke()
                .getColor(),
        ).toBe(
            "rgba(20, 20, 20, 0.5)",
        );

        expect(
            style
                .getText()
                .getFill()
                .getColor(),
        ).toBe(
            "rgba(20, 20, 20, 1)",
        );

        expect(
            style
                .getText()
                .getStroke()
                .getColor(),
        ).toBe(
            "rgba(255, 255, 255, 1)",
        );
    });

    it("updates opacity and the displayed opacity value", () => {
        const {
            map,
            graticule,
            gridOpacityInput,
            gridOpacityValue,
        } = createHarness({
            gridTheme:
                "light-contrast",
        });

        map.renderSync.mockClear();

        gridOpacityInput.value =
            "25";

        dispatchInput(
            gridOpacityInput,
        );

        expect(
            getGridStyle(
                graticule,
            )
                .getStroke()
                .getColor(),
        ).toBe(
            "rgba(0, 170, 200, 0.25)",
        );

        expect(
            gridOpacityValue.textContent,
        ).toBe("25%");

        expect(
            map.renderSync,
        ).toHaveBeenCalledOnce();
    });
});

describe("labels", () => {
    it("hides labels when label visibility is disabled", () => {
        const {
            map,
            graticule,
            gridLabelsVisibleInput,
        } = createHarness();

        map.renderSync.mockClear();

        gridLabelsVisibleInput.checked =
            false;

        dispatchChange(
            gridLabelsVisibleInput,
        );

        expect(
            getGridStyle(
                graticule,
            ).getText(),
        ).toBeNull();

        expect(
            map.renderSync,
        ).toHaveBeenCalledOnce();
    });

    it("restores labels when label visibility is enabled", () => {
        const {
            map,
            graticule,
            gridLabelsVisibleInput,
        } = createHarness({
            labelsVisible: false,
        });

        map.renderSync.mockClear();

        gridLabelsVisibleInput.checked =
            true;

        dispatchChange(
            gridLabelsVisibleInput,
        );

        expect(
            getGridStyle(
                graticule,
            ).getText(),
        ).not.toBeNull();

        expect(
            map.renderSync,
        ).toHaveBeenCalledOnce();
    });
});

describe("grid toggles", () => {
    it("activates the graticule and deactivates the screen-space grid", () => {
        const {
            map,
            graticule,
            screenSpaceGraticule,
            graticuleToggle,
            screenSpaceGraticuleToggle,
        } = createHarness();

        const mapState =
            stubGraticuleSetMap(
                graticule,
            );

        clickToggle(
            screenSpaceGraticuleToggle,
        );

        expect(
            mapState.getAssignedMap(
                screenSpaceGraticule,
            ),
        ).toBe(map);

        clickToggle(
            graticuleToggle,
        );

        expect(
            graticuleToggle.getActive(),
        ).toBe(true);

        expect(
            graticuleToggle
                .element.classList
                .contains("active"),
        ).toBe(true);

        expect(
            mapState.getAssignedMap(
                graticule,
            ),
        ).toBe(map);

        expect(
            screenSpaceGraticuleToggle
                .getActive(),
        ).toBe(false);

        expect(
            mapState.getAssignedMap(
                screenSpaceGraticule,
            ),
        ).toBeNull();
    });

    it("activates the screen-space grid and deactivates the graticule", () => {
        const {
            map,
            graticule,
            screenSpaceGraticule,
            graticuleToggle,
            screenSpaceGraticuleToggle,
        } = createHarness();

        const mapState =
            stubGraticuleSetMap(
                graticule,
            );

        clickToggle(
            graticuleToggle,
        );

        clickToggle(
            screenSpaceGraticuleToggle,
        );

        expect(
            screenSpaceGraticuleToggle
                .getActive(),
        ).toBe(true);

        expect(
            screenSpaceGraticuleToggle
                .element.classList
                .contains("active"),
        ).toBe(true);

        expect(
            mapState.getAssignedMap(
                screenSpaceGraticule,
            ),
        ).toBe(map);

        expect(
            graticuleToggle.getActive(),
        ).toBe(false);

        expect(
            mapState.getAssignedMap(
                graticule,
            ),
        ).toBeNull();
    });

    it("removes a grid when its active toggle is clicked again", () => {
        const {
            graticule,
            graticuleToggle,
        } = createHarness();

        const mapState =
            stubGraticuleSetMap(
                graticule,
            );

        clickToggle(
            graticuleToggle,
        );

        expect(
            graticuleToggle.getActive(),
        ).toBe(true);

        clickToggle(
            graticuleToggle,
        );

        expect(
            graticuleToggle.getActive(),
        ).toBe(false);

        expect(
            mapState.getAssignedMap(
                graticule,
            ),
        ).toBeNull();
    });
});

describe("grid recreation", () => {
    it.each([
        "fine",
        "default",
        "coarse",
        "unknown",
    ])(
        "recreates both grids when spacing changes to %s",
        (spacing) => {
            const {
                map,
                graticule,
                screenSpaceGraticule,
                gridSpacingSelect,
                onGraticulesChange,
            } = createHarness();

            const mapState =
                stubGraticuleSetMap(
                    graticule,
                );

            onGraticulesChange.mockClear();
            map.renderSync.mockClear();

            setSelectValue(
                gridSpacingSelect,
                spacing,
            );

            dispatchChange(
                gridSpacingSelect,
            );

            expect(
                onGraticulesChange,
            ).toHaveBeenCalledOnce();

            const [
                newGraticule,
                newScreenSpaceGraticule,
            ] =
                onGraticulesChange
                    .mock.calls[0];

            expect(
                newGraticule,
            ).not.toBe(graticule);

            expect(
                newScreenSpaceGraticule,
            ).not.toBe(
                screenSpaceGraticule,
            );

            expect(
                mapState.getAssignedMap(
                    graticule,
                ),
            ).toBeNull();

            expect(
                mapState.getAssignedMap(
                    screenSpaceGraticule,
                ),
            ).toBeNull();

            expect(
                map.renderSync,
            ).toHaveBeenCalledOnce();
        },
    );

    it("preserves an active graticule when spacing changes", () => {
        const {
            map,
            graticule,
            graticuleToggle,
            gridSpacingSelect,
            onGraticulesChange,
        } = createHarness();

        const mapState =
            stubGraticuleSetMap(
                graticule,
            );

        clickToggle(
            graticuleToggle,
        );

        onGraticulesChange.mockClear();
        map.renderSync.mockClear();

        gridSpacingSelect.value =
            "coarse";

        dispatchChange(
            gridSpacingSelect,
        );

        const [
            newGraticule,
            newScreenSpaceGraticule,
        ] =
            onGraticulesChange
                .mock.calls[0];

        expect(
            graticuleToggle.getActive(),
        ).toBe(true);

        expect(
            mapState.getAssignedMap(
                newGraticule,
            ),
        ).toBe(map);

        expect(
            mapState.getAssignedMap(
                newScreenSpaceGraticule,
            ),
        ).toBeNull();

        expect(
            map.renderSync,
        ).toHaveBeenCalledOnce();
    });

    it("preserves an active screen-space grid when projection changes", () => {
        const {
            map,
            controller,
            graticule,
            screenSpaceGraticuleToggle,
            onGraticulesChange,
        } = createHarness();

        const mapState =
            stubGraticuleSetMap(
                graticule,
            );

        clickToggle(
            screenSpaceGraticuleToggle,
        );

        onGraticulesChange.mockClear();
        map.renderSync.mockClear();

        controller.setProjection(
            createProjection(
                "EPSG:4326",
            ),
        );

        const [
            newGraticule,
            newScreenSpaceGraticule,
        ] =
            onGraticulesChange
                .mock.calls[0];

        expect(
            screenSpaceGraticuleToggle
                .getActive(),
        ).toBe(true);

        expect(
            mapState.getAssignedMap(
                newGraticule,
            ),
        ).toBeNull();

        expect(
            mapState.getAssignedMap(
                newScreenSpaceGraticule,
            ),
        ).toBe(map);

        expect(
            map.renderSync,
        ).toHaveBeenCalledOnce();
    });

    it("can recreate grids without preserving active state", () => {
        const {
            controller,
            graticule,
            graticuleToggle,
            screenSpaceGraticuleToggle,
            onGraticulesChange,
        } = createHarness();

        const mapState =
            stubGraticuleSetMap(
                graticule,
            );

        clickToggle(
            graticuleToggle,
        );

        onGraticulesChange.mockClear();

        controller.setProjection(
            createProjection(
                "EPSG:4326",
            ),
            {
                preserveActive: false,
            },
        );

        const [
            newGraticule,
            newScreenSpaceGraticule,
        ] =
            onGraticulesChange
                .mock.calls[0];

        expect(
            graticuleToggle.getActive(),
        ).toBe(false);

        expect(
            screenSpaceGraticuleToggle
                .getActive(),
        ).toBe(false);

        expect(
            graticuleToggle
                .element.classList
                .contains("active"),
        ).toBe(false);

        expect(
            screenSpaceGraticuleToggle
                .element.classList
                .contains("active"),
        ).toBe(false);

        expect(
            mapState.getAssignedMap(
                newGraticule,
            ),
        ).toBeNull();

        expect(
            mapState.getAssignedMap(
                newScreenSpaceGraticule,
            ),
        ).toBeNull();
    });
});

describe("visibility and viewer state", () => {
    it("hides disabled grid controls and deactivates their grids", () => {
        const {
            graticule,
            screenSpaceGraticule,
            graticuleToggle,
            screenSpaceGraticuleToggle,
            graticuleVisibleInput,
            screenSpaceGraticuleVisibleInput,
            controller,
        } = createHarness();

        const mapState =
            stubGraticuleSetMap(
                graticule,
            );

        clickToggle(
            graticuleToggle,
        );

        graticuleVisibleInput.checked =
            false;

        screenSpaceGraticuleVisibleInput.checked =
            false;

        controller.updateVisibility();

        expect(
            graticuleToggle
                .element.classList
                .contains(
                    "viewer-control-hidden",
                ),
        ).toBe(true);

        expect(
            screenSpaceGraticuleToggle
                .element.classList
                .contains(
                    "viewer-control-hidden",
                ),
        ).toBe(true);

        expect(
            graticuleToggle.getActive(),
        ).toBe(false);

        expect(
            screenSpaceGraticuleToggle
                .getActive(),
        ).toBe(false);

        expect(
            mapState.getAssignedMap(
                graticule,
            ),
        ).toBeNull();

        expect(
            mapState.getAssignedMap(
                screenSpaceGraticule,
            ),
        ).toBeNull();
    });

    it("shows grid controls when their visibility settings are enabled", () => {
        const harness =
            createHarness({
                graticuleVisible: false,
                screenSpaceGraticuleVisible:
                    false,
            });

        harness.controller
            .updateVisibility();

        harness.graticuleVisibleInput
            .checked = true;

        harness
            .screenSpaceGraticuleVisibleInput
            .checked = true;

        harness.controller
            .updateVisibility();

        expect(
            harness.graticuleToggle
                .element.classList
                .contains(
                    "viewer-control-hidden",
                ),
        ).toBe(false);

        expect(
            harness
                .screenSpaceGraticuleToggle
                .element.classList
                .contains(
                    "viewer-control-hidden",
                ),
        ).toBe(false);
    });

    it("disables both grid controls and clears active grids when the viewer is disabled", () => {
        const {
            graticule,
            screenSpaceGraticule,
            graticuleToggle,
            screenSpaceGraticuleToggle,
            controller,
        } = createHarness();

        const mapState =
            stubGraticuleSetMap(
                graticule,
            );

        clickToggle(
            screenSpaceGraticuleToggle,
        );

        controller.setViewerEnabled(
            false,
        );

        expect(
            graticuleToggle
                .element.querySelector(
                    "button",
                ).disabled,
        ).toBe(true);

        expect(
            screenSpaceGraticuleToggle
                .element.querySelector(
                    "button",
                ).disabled,
        ).toBe(true);

        expect(
            graticuleToggle.getActive(),
        ).toBe(false);

        expect(
            screenSpaceGraticuleToggle
                .getActive(),
        ).toBe(false);

        expect(
            mapState.getAssignedMap(
                graticule,
            ),
        ).toBeNull();

        expect(
            mapState.getAssignedMap(
                screenSpaceGraticule,
            ),
        ).toBeNull();
    });

    it("reenables both grid buttons without reactivating a grid", () => {
        const {
            graticule,
            graticuleToggle,
            screenSpaceGraticuleToggle,
            controller,
        } = createHarness();

        stubGraticuleSetMap(
            graticule,
        );

        clickToggle(
            graticuleToggle,
        );

        controller.setViewerEnabled(
            false,
        );

        controller.setViewerEnabled(
            true,
        );

        expect(
            graticuleToggle
                .element.querySelector(
                    "button",
                ).disabled,
        ).toBe(false);

        expect(
            screenSpaceGraticuleToggle
                .element.querySelector(
                    "button",
                ).disabled,
        ).toBe(false);

        expect(
            graticuleToggle.getActive(),
        ).toBe(false);

        expect(
            screenSpaceGraticuleToggle
                .getActive(),
        ).toBe(false);
    });
});
