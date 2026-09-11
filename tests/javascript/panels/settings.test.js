import {
    afterEach,
    beforeEach,
    describe,
    expect,
    it,
    vi,
} from "vitest";

import {
    createSettingsPanelController,
} from "../../../tiatoolbox/visualization/openlayers/src/panels/settings.js";

const settingsStorageKey =
    "tiatoolbox-openlayers-settings";

function createCheckbox(
    checked = true,
) {
    const input =
        document.createElement("input");

    input.type = "checkbox";
    input.checked = checked;

    return input;
}

function createInput(
    value,
    type = "range",
) {
    const input =
        document.createElement("input");

    input.type = type;

    if (type === "range") {
        input.min = "0";
        input.max = "100";
    }

    input.value = value;

    return input;
}

function createSelect(
    values,
    value,
) {
    const select =
        document.createElement("select");

    for (const optionValue of values) {
        const option =
            document.createElement("option");

        option.value = optionValue;
        option.textContent =
            optionValue;

        select.append(option);
    }

    select.value = value;

    return select;
}

function createHarness() {
    const viewerApp =
        document.createElement("div");

    const panel =
        document.createElement("div");

    panel.className = "hidden";

    const toggle =
        document.createElement("button");

    const closeButton =
        document.createElement("button");

    const resetDefaultsButton =
        document.createElement("button");

    const interfaceTab =
        document.createElement("button");

    interfaceTab.dataset.settingsTab =
        "interface";
    interfaceTab.classList.add(
        "active",
    );

    const controlsTab =
        document.createElement("button");

    controlsTab.dataset.settingsTab =
        "controls";

    const interfacePanel =
        document.createElement("div");

    interfacePanel.dataset.settingsPanel =
        "interface";

    const controlsPanel =
        document.createElement("div");

    controlsPanel.dataset.settingsPanel =
        "controls";
    controlsPanel.classList.add(
        "hidden",
    );

    const tabs = [
        interfaceTab,
        controlsTab,
    ];

    const tabPanels = [
        interfacePanel,
        controlsPanel,
    ];

    const themeSelect =
        createSelect(
            [
                "dark",
                "light",
                "high-contrast",
            ],
            "dark",
        );

    const controlOpacityInput =
        createInput("100");

    const controlOpacityValue =
        document.createElement("span");

    const zoomVisibleInput =
        createCheckbox();

    const zoomLevelVisibleInput =
        createCheckbox();

    const rotationVisibleInput =
        createCheckbox();

    const graticuleVisibleInput =
        createCheckbox();

    const screenSpaceGraticuleVisibleInput =
        createCheckbox();

    const resetViewVisibleInput =
        createCheckbox();

    const fullscreenVisibleInput =
        createCheckbox();

    const mousePositionVisibleInput =
        createCheckbox();

    const overviewMapVisibleInput =
        createCheckbox();

    const overviewMapSizeSelect =
        createSelect(
            [
                "small",
                "default",
                "large",
            ],
            "default",
        );

    const mouseWheelZoomSensitivitySelect =
        createSelect(
            [
                "low",
                "default",
                "high",
            ],
            "default",
        );

    const zoomButtonStepSelect =
        createSelect(
            [
                "0.1",
                "0.5",
                "1",
                "2",
            ],
            "1",
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
            "default",
        );

    const gridOpacityInput =
        createInput("50");

    const gridSpacingSelect =
        createSelect(
            [
                "fine",
                "default",
                "coarse",
            ],
            "default",
        );

    const gridLabelsVisibleInput =
        createCheckbox();

    const scaleBarEnabledInput =
        createCheckbox();

    const scaleBarColourInput =
        createInput(
            "#ffffff",
            "color",
        );

    const scaleBarOpacityInput =
        createInput("100");

    const scaleBarSizeSelect =
        createSelect(
            [
                "small",
                "default",
                "large",
            ],
            "default",
        );

    const scaleBarUnitsSelect =
        createSelect(
            [
                "metric",
                "imperial",
            ],
            "metric",
        );

    const onThemeChange = vi.fn();

    const onControlVisibilityChange =
        vi.fn();

    const onReset = vi.fn();

    panel.append(
        closeButton,
        interfaceTab,
        controlsTab,
        interfacePanel,
        controlsPanel,
        resetDefaultsButton,
    );

    document.body.append(
        viewerApp,
        toggle,
        panel,
    );

    const controller =
        createSettingsPanelController({
            viewerApp,
            panel,
            toggle,
            closeButton,
            tabs,
            tabPanels,
            resetDefaultsButton,
            themeSelect,
            controlOpacityInput,
            controlOpacityValue,
            zoomVisibleInput,
            zoomLevelVisibleInput,
            rotationVisibleInput,
            graticuleVisibleInput,
            screenSpaceGraticuleVisibleInput,
            resetViewVisibleInput,
            fullscreenVisibleInput,
            mousePositionVisibleInput,
            overviewMapVisibleInput,
            overviewMapSizeSelect,
            mouseWheelZoomSensitivitySelect,
            zoomButtonStepSelect,
            gridThemeSelect,
            gridOpacityInput,
            gridSpacingSelect,
            gridLabelsVisibleInput,
            scaleBarEnabledInput,
            scaleBarColourInput,
            scaleBarOpacityInput,
            scaleBarSizeSelect,
            scaleBarUnitsSelect,
            onThemeChange,
            onControlVisibilityChange,
            onReset,
        });

    return {
        viewerApp,
        panel,
        toggle,
        closeButton,
        resetDefaultsButton,
        tabs,
        tabPanels,
        themeSelect,
        controlOpacityInput,
        controlOpacityValue,
        zoomVisibleInput,
        zoomLevelVisibleInput,
        rotationVisibleInput,
        graticuleVisibleInput,
        screenSpaceGraticuleVisibleInput,
        resetViewVisibleInput,
        fullscreenVisibleInput,
        mousePositionVisibleInput,
        overviewMapVisibleInput,
        overviewMapSizeSelect,
        mouseWheelZoomSensitivitySelect,
        zoomButtonStepSelect,
        gridThemeSelect,
        gridOpacityInput,
        gridSpacingSelect,
        gridLabelsVisibleInput,
        scaleBarEnabledInput,
        scaleBarColourInput,
        scaleBarOpacityInput,
        scaleBarSizeSelect,
        scaleBarUnitsSelect,
        onThemeChange,
        onControlVisibilityChange,
        onReset,
        controller,
    };
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

beforeEach(() => {
    document.body.replaceChildren();
    window.localStorage.clear();
});

afterEach(() => {
    document.body.replaceChildren();
    window.localStorage.clear();
    vi.restoreAllMocks();
});

describe("panel state and tabs", () => {
    it("opens and closes through the controller", () => {
        // Test opening and closing the settings panel through its controller.
        const {
            panel,
            toggle,
            controller,
        } = createHarness();

        controller.setOpen(true);

        expect(
            panel.classList.contains(
                "hidden",
            ),
        ).toBe(false);

        expect(
            toggle.classList.contains(
                "active",
            ),
        ).toBe(true);

        controller.setOpen(false);

        expect(
            panel.classList.contains(
                "hidden",
            ),
        ).toBe(true);

        expect(
            toggle.classList.contains(
                "active",
            ),
        ).toBe(false);
    });

    it("opens from the toggle and closes from the close button", () => {
        // Test the settings buttons open and close the panel.
        const {
            panel,
            toggle,
            closeButton,
            controller,
        } = createHarness();

        controller.bindEvents();

        toggle.click();

        expect(
            panel.classList.contains(
                "hidden",
            ),
        ).toBe(false);

        closeButton.click();

        expect(
            panel.classList.contains(
                "hidden",
            ),
        ).toBe(true);
    });

    it("switches between settings tabs", () => {
        // Test switching between settings tabs.
        const {
            tabs,
            tabPanels,
            controller,
        } = createHarness();

        controller.bindEvents();

        tabs[1].click();

        expect(
            tabs[0].classList.contains(
                "active",
            ),
        ).toBe(false);

        expect(
            tabs[1].classList.contains(
                "active",
            ),
        ).toBe(true);

        expect(
            tabPanels[0].classList.contains(
                "hidden",
            ),
        ).toBe(true);

        expect(
            tabPanels[1].classList.contains(
                "hidden",
            ),
        ).toBe(false);
    });

    it("only binds events once", () => {
        // Test settings event handlers are only bound once.
        const {
            zoomVisibleInput,
            onControlVisibilityChange,
            controller,
        } = createHarness();

        controller.bindEvents();
        controller.bindEvents();

        zoomVisibleInput.checked = false;

        dispatchChange(
            zoomVisibleInput,
        );

        expect(
            onControlVisibilityChange,
        ).toHaveBeenCalledOnce();
    });
});

describe("interface appearance", () => {
    it("updates the dark interface appearance", () => {
        // Test applying the dark interface appearance.
        const {
            viewerApp,
            controlOpacityInput,
            controlOpacityValue,
            controller,
        } = createHarness();

        controlOpacityInput.value = "80";

        controller.updateAppearance();

        expect(
            viewerApp.style.getPropertyValue(
                "--viewer-control-background",
            ),
        ).toBe(
            "rgba(17, 17, 17, 0.8)",
        );

        expect(
            viewerApp.style.getPropertyValue(
                "--viewer-control-foreground",
            ),
        ).toBe("#ffffff");

        expect(
            viewerApp.style.getPropertyValue(
                "--viewer-control-muted-foreground",
            ),
        ).toBe(
            "rgba(255, 255, 255, 0.7)",
        );

        expect(
            controlOpacityValue.textContent,
        ).toBe("80%");
    });

    it("updates appearance, calls the theme callback, and saves on theme change", () => {
        // Test changing theme updates appearance, applies the theme, and saves it.
        const {
            viewerApp,
            themeSelect,
            onThemeChange,
            controller,
        } = createHarness();

        controller.bindEvents();

        themeSelect.value = "light";

        dispatchChange(
            themeSelect,
        );

        expect(
            viewerApp.style.getPropertyValue(
                "--viewer-control-background",
            ),
        ).toBe(
            "rgba(242, 242, 242, 1)",
        );

        expect(
            viewerApp.style.getPropertyValue(
                "--viewer-control-foreground",
            ),
        ).toBe("#000000");

        expect(
            onThemeChange,
        ).toHaveBeenCalledOnce();

        const saved = JSON.parse(
            window.localStorage.getItem(
                settingsStorageKey,
            ),
        );

        expect(saved.theme).toBe(
            "light",
        );
    });

    it("updates and saves interface opacity on input", () => {
        // Test changing interface opacity updates and saves the value.
        const {
            viewerApp,
            controlOpacityInput,
            controlOpacityValue,
            controller,
        } = createHarness();

        controller.bindEvents();

        controlOpacityInput.value = "65";

        dispatchInput(
            controlOpacityInput,
        );

        expect(
            viewerApp.style.getPropertyValue(
                "--viewer-control-background",
            ),
        ).toBe(
            "rgba(17, 17, 17, 0.65)",
        );

        expect(
            controlOpacityValue.textContent,
        ).toBe("65%");

        const saved = JSON.parse(
            window.localStorage.getItem(
                settingsStorageKey,
            ),
        );

        expect(
            saved.interfaceOpacity,
        ).toBe("65");
    });
});

describe("saving settings", () => {
    it("saves the complete current settings state", () => {
        // Test saving includes the complete current settings state.
        const harness =
            createHarness();

        harness.controller.bindEvents();

        harness.themeSelect.value =
            "high-contrast";

        harness.controlOpacityInput.value =
            "75";

        harness.zoomVisibleInput.checked =
            false;

        harness.zoomLevelVisibleInput.checked =
            false;

        harness.rotationVisibleInput.checked =
            false;

        harness.graticuleVisibleInput.checked =
            false;

        harness.screenSpaceGraticuleVisibleInput.checked =
            false;

        harness.resetViewVisibleInput.checked =
            false;

        harness.fullscreenVisibleInput.checked =
            false;

        harness.mousePositionVisibleInput.checked =
            false;

        harness.overviewMapVisibleInput.checked =
            false;

        harness.mouseWheelZoomSensitivitySelect.value =
            "high";

        harness.zoomButtonStepSelect.value =
            "0.5";

        harness.overviewMapSizeSelect.value =
            "large";

        harness.gridThemeSelect.value =
            "dark-contrast";

        harness.gridOpacityInput.value =
            "25";

        harness.gridSpacingSelect.value =
            "coarse";

        harness.gridLabelsVisibleInput.checked =
            false;

        harness.scaleBarEnabledInput.checked =
            false;

        harness.scaleBarColourInput.value =
            "#123456";

        harness.scaleBarOpacityInput.value =
            "35";

        harness.scaleBarSizeSelect.value =
            "large";

        harness.scaleBarUnitsSelect.value =
            "imperial";

        dispatchChange(
            harness.zoomButtonStepSelect,
        );

        expect(
            JSON.parse(
                window.localStorage.getItem(
                    settingsStorageKey,
                ),
            ),
        ).toEqual({
            theme: "high-contrast",
            interfaceOpacity: "75",

            controls: {
                zoom: false,
                zoomLevel: false,
                rotation: false,
                graticule: false,
                screenSpaceGraticule:
                    false,
                resetView: false,
                fullscreen: false,
                mousePosition: false,
                overviewMap: false,
            },

            navigation: {
                mouseWheelZoomSensitivity:
                    "high",
                zoomButtonStep: "0.5",
            },

            overviewMap: {
                size: "large",
            },

            grid: {
                theme: "dark-contrast",
                opacity: "25",
                spacing: "coarse",
                labels: false,
            },

            scaleBar: {
                enabled: false,
                colour: "#123456",
                opacity: "35",
                size: "large",
                units: "imperial",
            },
        });
    });

    it("saves changes from visibility controls", () => {
        // Test control visibility changes are saved.
        const {
            zoomVisibleInput,
            onControlVisibilityChange,
            controller,
        } = createHarness();

        controller.bindEvents();

        zoomVisibleInput.checked = false;

        dispatchChange(
            zoomVisibleInput,
        );

        expect(
            onControlVisibilityChange,
        ).toHaveBeenCalledOnce();

        const saved = JSON.parse(
            window.localStorage.getItem(
                settingsStorageKey,
            ),
        );

        expect(
            saved.controls.zoom,
        ).toBe(false);
    });

    it("saves changes from grid and scale bar inputs", () => {
        // Test grid and scale bar changes are saved.
        const {
            gridOpacityInput,
            scaleBarColourInput,
            controller,
        } = createHarness();

        controller.bindEvents();

        gridOpacityInput.value = "20";
        dispatchInput(
            gridOpacityInput,
        );

        let saved = JSON.parse(
            window.localStorage.getItem(
                settingsStorageKey,
            ),
        );

        expect(
            saved.grid.opacity,
        ).toBe("20");

        scaleBarColourInput.value =
            "#abcdef";

        dispatchInput(
            scaleBarColourInput,
        );

        saved = JSON.parse(
            window.localStorage.getItem(
                settingsStorageKey,
            ),
        );

        expect(
            saved.scaleBar.colour,
        ).toBe("#abcdef");
    });

    it("continues when local storage cannot save settings", () => {
        // Test settings continue to work when saving to local storage fails.
        const {
            gridThemeSelect,
            controller,
        } = createHarness();

        vi.spyOn(
            Storage.prototype,
            "setItem",
        ).mockImplementation(
            () => {
                throw new Error(
                    "Storage unavailable",
                );
            },
        );

        controller.bindEvents();

        expect(() => {
            dispatchChange(
                gridThemeSelect,
            );
        }).not.toThrow();
    });
});

describe("loading settings", () => {
    it("returns safely when no saved settings exist", () => {
        // Test loading settings safely handles an empty local storage.
        const {
            controller,
        } = createHarness();

        expect(() => {
            controller.load();
        }).not.toThrow();
    });

    it("loads valid saved settings", () => {
        // Test valid saved settings are restored to the controls.
        const harness =
            createHarness();

        window.localStorage.setItem(
            settingsStorageKey,
            JSON.stringify({
                theme: "light",
                interfaceOpacity: "65",

                controls: {
                    zoom: false,
                    zoomLevel: true,
                    rotation: false,
                    graticule: true,
                    screenSpaceGraticule:
                        false,
                    resetView: true,
                    fullscreen: false,
                    mousePosition: true,
                    overviewMap: false,
                },

                navigation: {
                    mouseWheelZoomSensitivity:
                        "high",
                    zoomButtonStep: "0.5",
                },

                overviewMap: {
                    size: "large",
                },

                grid: {
                    theme: "dark-contrast",
                    opacity: "25",
                    spacing: "coarse",
                    labels: false,
                },

                scaleBar: {
                    enabled: false,
                    colour: "#123456",
                    opacity: "35",
                    size: "large",
                    units: "imperial",
                },
            }),
        );

        harness.controller.load();

        expect(
            harness.themeSelect.value,
        ).toBe("light");

        expect(
            harness.controlOpacityInput.value,
        ).toBe("65");

        expect(
            harness.zoomVisibleInput.checked,
        ).toBe(false);

        expect(
            harness.zoomLevelVisibleInput.checked,
        ).toBe(true);

        expect(
            harness.rotationVisibleInput.checked,
        ).toBe(false);

        expect(
            harness.graticuleVisibleInput.checked,
        ).toBe(true);

        expect(
            harness.screenSpaceGraticuleVisibleInput.checked,
        ).toBe(false);

        expect(
            harness.resetViewVisibleInput.checked,
        ).toBe(true);

        expect(
            harness.fullscreenVisibleInput.checked,
        ).toBe(false);

        expect(
            harness.mousePositionVisibleInput.checked,
        ).toBe(true);

        expect(
            harness.overviewMapVisibleInput.checked,
        ).toBe(false);

        expect(
            harness.mouseWheelZoomSensitivitySelect.value,
        ).toBe("high");

        expect(
            harness.zoomButtonStepSelect.value,
        ).toBe("0.5");

        expect(
            harness.overviewMapSizeSelect.value,
        ).toBe("large");

        expect(
            harness.gridThemeSelect.value,
        ).toBe("dark-contrast");

        expect(
            harness.gridOpacityInput.value,
        ).toBe("25");

        expect(
            harness.gridSpacingSelect.value,
        ).toBe("coarse");

        expect(
            harness.gridLabelsVisibleInput.checked,
        ).toBe(false);

        expect(
            harness.scaleBarEnabledInput.checked,
        ).toBe(false);

        expect(
            harness.scaleBarColourInput.value,
        ).toBe("#123456");

        expect(
            harness.scaleBarOpacityInput.value,
        ).toBe("35");

        expect(
            harness.scaleBarSizeSelect.value,
        ).toBe("large");

        expect(
            harness.scaleBarUnitsSelect.value,
        ).toBe("imperial");
    });

    it("ignores invalid saved setting values", () => {
        // Test invalid saved setting values are ignored.
        const harness =
            createHarness();

        window.localStorage.setItem(
            settingsStorageKey,
            JSON.stringify({
                theme: "invalid",
                interfaceOpacity: "33",

                controls: {
                    zoom: "false",
                    zoomLevel: 0,
                    rotation: null,
                    graticule: "true",
                    screenSpaceGraticule:
                        1,
                    resetView: "yes",
                    fullscreen: {},
                    mousePosition: [],
                    overviewMap: "false",
                },

                navigation: {
                    mouseWheelZoomSensitivity:
                        "extreme",
                    zoomButtonStep: "3",
                },

                overviewMap: {
                    size: "huge",
                },

                grid: {
                    theme: "invalid",
                    opacity: "101",
                    spacing: "invalid",
                    labels: "false",
                },

                scaleBar: {
                    enabled: "true",
                    colour: "white",
                    opacity: "-1",
                    size: "huge",
                    units: "unknown",
                },
            }),
        );

        harness.controller.load();

        expect(
            harness.themeSelect.value,
        ).toBe("dark");

        expect(
            harness.controlOpacityInput.value,
        ).toBe("100");

        for (const input of [
            harness.zoomVisibleInput,
            harness.zoomLevelVisibleInput,
            harness.rotationVisibleInput,
            harness.graticuleVisibleInput,
            harness.screenSpaceGraticuleVisibleInput,
            harness.resetViewVisibleInput,
            harness.fullscreenVisibleInput,
            harness.mousePositionVisibleInput,
            harness.overviewMapVisibleInput,
        ]) {
            expect(
                input.checked,
            ).toBe(true);
        }

        expect(
            harness.mouseWheelZoomSensitivitySelect.value,
        ).toBe("default");

        expect(
            harness.zoomButtonStepSelect.value,
        ).toBe("1");

        expect(
            harness.overviewMapSizeSelect.value,
        ).toBe("default");

        expect(
            harness.gridThemeSelect.value,
        ).toBe("default");

        expect(
            harness.gridOpacityInput.value,
        ).toBe("50");

        expect(
            harness.gridSpacingSelect.value,
        ).toBe("default");

        expect(
            harness.gridLabelsVisibleInput.checked,
        ).toBe(true);

        expect(
            harness.scaleBarEnabledInput.checked,
        ).toBe(true);

        expect(
            harness.scaleBarColourInput.value,
        ).toBe("#ffffff");

        expect(
            harness.scaleBarOpacityInput.value,
        ).toBe("100");

        expect(
            harness.scaleBarSizeSelect.value,
        ).toBe("default");

        expect(
            harness.scaleBarUnitsSelect.value,
        ).toBe("metric");
    });

    it.each([
        "{",
        "null",
        "42",
        '"settings"',
    ])(
        "ignores unusable stored data %s",
        (storedValue) => {
            // Test unusable saved settings data is ignored.
            const harness =
                createHarness();

            window.localStorage.setItem(
                settingsStorageKey,
                storedValue,
            );

            expect(() => {
                harness.controller.load();
            }).not.toThrow();

            expect(
                harness.themeSelect.value,
            ).toBe("dark");
        },
    );

    it("continues when local storage cannot be read", () => {
        // Test settings continue to work when local storage cannot be read.
        const {
            controller,
        } = createHarness();

        vi.spyOn(
            Storage.prototype,
            "getItem",
        ).mockImplementation(
            () => {
                throw new Error(
                    "Storage unavailable",
                );
            },
        );

        expect(() => {
            controller.load();
        }).not.toThrow();
    });
});

describe("resetting settings", () => {
    it("restores all default values", () => {
        // Test resetting settings restores all default values.
        const harness =
            createHarness();

        harness.themeSelect.value =
            "light";

        harness.controlOpacityInput.value =
            "40";

        for (const input of [
            harness.zoomVisibleInput,
            harness.zoomLevelVisibleInput,
            harness.rotationVisibleInput,
            harness.graticuleVisibleInput,
            harness.screenSpaceGraticuleVisibleInput,
            harness.resetViewVisibleInput,
            harness.fullscreenVisibleInput,
            harness.mousePositionVisibleInput,
            harness.overviewMapVisibleInput,
        ]) {
            input.checked = false;
        }

        harness.overviewMapSizeSelect.value =
            "large";

        harness.mouseWheelZoomSensitivitySelect.value =
            "high";

        harness.zoomButtonStepSelect.value =
            "2";

        harness.gridThemeSelect.value =
            "dark";

        harness.gridOpacityInput.value =
            "10";

        harness.gridSpacingSelect.value =
            "coarse";

        harness.gridLabelsVisibleInput.checked =
            false;

        harness.scaleBarEnabledInput.checked =
            false;

        harness.scaleBarColourInput.value =
            "#123456";

        harness.scaleBarOpacityInput.value =
            "20";

        harness.scaleBarSizeSelect.value =
            "large";

        harness.scaleBarUnitsSelect.value =
            "imperial";

        harness.controller.resetValues();

        expect(
            harness.themeSelect.value,
        ).toBe("dark");

        expect(
            harness.controlOpacityInput.value,
        ).toBe("100");

        for (const input of [
            harness.zoomVisibleInput,
            harness.zoomLevelVisibleInput,
            harness.rotationVisibleInput,
            harness.graticuleVisibleInput,
            harness.screenSpaceGraticuleVisibleInput,
            harness.resetViewVisibleInput,
            harness.fullscreenVisibleInput,
            harness.mousePositionVisibleInput,
            harness.overviewMapVisibleInput,
        ]) {
            expect(
                input.checked,
            ).toBe(true);
        }

        expect(
            harness.overviewMapSizeSelect.value,
        ).toBe("default");

        expect(
            harness.mouseWheelZoomSensitivitySelect.value,
        ).toBe("default");

        expect(
            harness.zoomButtonStepSelect.value,
        ).toBe("1");

        expect(
            harness.gridThemeSelect.value,
        ).toBe("default");

        expect(
            harness.gridOpacityInput.value,
        ).toBe("50");

        expect(
            harness.gridSpacingSelect.value,
        ).toBe("default");

        expect(
            harness.gridLabelsVisibleInput.checked,
        ).toBe(true);

        expect(
            harness.scaleBarEnabledInput.checked,
        ).toBe(true);

        expect(
            harness.scaleBarColourInput.value,
        ).toBe("#ffffff");

        expect(
            harness.scaleBarOpacityInput.value,
        ).toBe("100");

        expect(
            harness.scaleBarSizeSelect.value,
        ).toBe("default");

        expect(
            harness.scaleBarUnitsSelect.value,
        ).toBe("metric");
    });

    it("clears saved settings", () => {
        // Test resetting settings clears saved local storage.
        const {
            controller,
        } = createHarness();

        window.localStorage.setItem(
            settingsStorageKey,
            "{}",
        );

        controller.clearSavedSettings();

        expect(
            window.localStorage.getItem(
                settingsStorageKey,
            ),
        ).toBeNull();
    });

    it("continues when saved settings cannot be cleared", () => {
        // Test reset still completes when local storage cannot be cleared.
        const {
            controller,
        } = createHarness();

        vi.spyOn(
            Storage.prototype,
            "removeItem",
        ).mockImplementation(
            () => {
                throw new Error(
                    "Storage unavailable",
                );
            },
        );

        expect(() => {
            controller.clearSavedSettings();
        }).not.toThrow();
    });

    it("delegates the reset button to the reset callback", () => {
        // Test the reset button calls the supplied reset callback.
        const {
            resetDefaultsButton,
            onReset,
            controller,
        } = createHarness();

        controller.bindEvents();

        resetDefaultsButton.click();

        expect(
            onReset,
        ).toHaveBeenCalledOnce();
    });
});
