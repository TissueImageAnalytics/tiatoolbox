import {
  getContrastingColour,
  hexToRgb,
  mixColour,
  toRgba,
} from "../utils/colours.js";

const interfaceThemeColours = {
  dark: "#111111",
  light: "#f2f2f2",
  "high-contrast": "#000000",
};

const settingsStorageKey =
  "tiatoolbox-openlayers-settings";

function createSettingsPanelController({
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
}) {
  let eventsBound = false;

  function setOpen(open) {
    panel.classList.toggle(
      "hidden",
      !open,
    );

    toggle.classList.toggle(
      "active",
      open,
    );
  }

  function updateAppearance() {
    const themeColour =
      interfaceThemeColours[
        themeSelect.value
      ] ??
      interfaceThemeColours.dark;

    const colour =
      hexToRgb(themeColour);

    if (colour === null) {
      return;
    }

    const opacity =
      Number(
        controlOpacityInput.value,
      ) / 100;

    const foreground =
      getContrastingColour(colour);

    const foregroundRgb =
      foreground === "#ffffff"
        ? {
            r: 255,
            g: 255,
            b: 255,
          }
        : {
            r: 0,
            g: 0,
            b: 0,
          };

    const shadowColour =
      foreground === "#ffffff"
        ? {
            r: 0,
            g: 0,
            b: 0,
          }
        : {
            r: 255,
            g: 255,
            b: 255,
          };

    const interactionTarget =
      foreground === "#ffffff"
        ? 255
        : 0;

    const surfaceColour = mixColour(
      colour,
      interactionTarget,
      0.08,
    );

    const hoverColour = mixColour(
      colour,
      interactionTarget,
      0.16,
    );

    const pressedColour = mixColour(
      colour,
      interactionTarget,
      0.28,
    );

    const borderColour = mixColour(
      colour,
      interactionTarget,
      0.4,
    );

    const focusBorderColour =
      mixColour(
        colour,
        interactionTarget,
        0.58,
      );

    viewerApp.style.setProperty(
      "--viewer-control-background",
      toRgba(colour, opacity),
    );

    viewerApp.style.setProperty(
      "--viewer-control-surface-background",
      toRgba(
        surfaceColour,
        opacity,
      ),
    );

    viewerApp.style.setProperty(
      "--viewer-control-hover-background",
      toRgba(
        hoverColour,
        opacity,
      ),
    );

    viewerApp.style.setProperty(
      "--viewer-control-pressed-background",
      toRgba(
        pressedColour,
        opacity,
      ),
    );

    viewerApp.style.setProperty(
      "--viewer-control-foreground",
      foreground,
    );

    viewerApp.style.setProperty(
      "--viewer-control-hover-foreground",
      getContrastingColour(
        hoverColour,
      ),
    );

    viewerApp.style.setProperty(
      "--viewer-control-pressed-foreground",
      getContrastingColour(
        pressedColour,
      ),
    );

    viewerApp.style.setProperty(
      "--viewer-control-muted-foreground",
      toRgba(
        foregroundRgb,
        0.7,
      ),
    );

    viewerApp.style.setProperty(
      "--viewer-control-subtle-foreground",
      toRgba(
        foregroundRgb,
        0.55,
      ),
    );

    viewerApp.style.setProperty(
      "--viewer-control-border",
      toRgba(
        borderColour,
        Math.max(
          opacity,
          0.7,
        ),
      ),
    );

    viewerApp.style.setProperty(
      "--viewer-control-focus-border",
      toRgba(
        focusBorderColour,
        Math.max(
          opacity,
          0.9,
        ),
      ),
    );

    viewerApp.style.setProperty(
      "--viewer-control-foreground-shadow",
      toRgba(
        shadowColour,
        0.75,
      ),
    );

    controlOpacityValue.textContent =
      `${controlOpacityInput.value}%`;
  }

  function save() {
    const settings = {
      theme: themeSelect.value,
      interfaceOpacity:
        controlOpacityInput.value,

      controls: {
        zoom: zoomVisibleInput.checked,
        zoomLevel:
          zoomLevelVisibleInput.checked,
        rotation:
          rotationVisibleInput.checked,
        graticule:
          graticuleVisibleInput.checked,
        screenSpaceGraticule:
          screenSpaceGraticuleVisibleInput.checked,
        resetView:
          resetViewVisibleInput.checked,
        fullscreen:
          fullscreenVisibleInput.checked,
        mousePosition:
          mousePositionVisibleInput.checked,
        overviewMap:
          overviewMapVisibleInput.checked,
      },

      navigation: {
        mouseWheelZoomSensitivity:
          mouseWheelZoomSensitivitySelect.value,
        zoomButtonStep:
          zoomButtonStepSelect.value,
      },

      overviewMap: {
        size: overviewMapSizeSelect.value,
      },

      grid: {
        theme: gridThemeSelect.value,
        opacity: gridOpacityInput.value,
        spacing: gridSpacingSelect.value,
        labels:
          gridLabelsVisibleInput.checked,
      },

      scaleBar: {
        enabled:
          scaleBarEnabledInput.checked,
        colour:
          scaleBarColourInput.value,
        opacity:
          scaleBarOpacityInput.value,
        size: scaleBarSizeSelect.value,
        units:
          scaleBarUnitsSelect.value,
      },
    };

    try {
      window.localStorage.setItem(
        settingsStorageKey,
        JSON.stringify(settings),
      );
    } catch {
      // Continue using the viewer if storage is unavailable.
    }
  }

  function load() {
    let savedSettings;

    try {
      const storedSettings =
        window.localStorage.getItem(
          settingsStorageKey,
        );

      if (storedSettings === null) {
        return;
      }

      savedSettings =
        JSON.parse(storedSettings);
    } catch {
      return;
    }

    if (
      savedSettings === null ||
      typeof savedSettings !== "object"
    ) {
      return;
    }

    if (
      [
        "dark",
        "light",
        "high-contrast",
      ].includes(savedSettings.theme)
    ) {
      themeSelect.value =
        savedSettings.theme;
    }

    if (
      [
        "40",
        "45",
        "50",
        "55",
        "60",
        "65",
        "70",
        "75",
        "80",
        "85",
        "90",
        "95",
        "100",
      ].includes(
        savedSettings.interfaceOpacity,
      )
    ) {
      controlOpacityInput.value =
        savedSettings.interfaceOpacity;
    }

    const controls =
      savedSettings.controls;

    if (
      controls !== null &&
      typeof controls === "object"
    ) {
      if (
        typeof controls.zoom ===
        "boolean"
      ) {
        zoomVisibleInput.checked =
          controls.zoom;
      }

      if (
        typeof controls.zoomLevel ===
        "boolean"
      ) {
        zoomLevelVisibleInput.checked =
          controls.zoomLevel;
      }

      if (
        typeof controls.rotation ===
        "boolean"
      ) {
        rotationVisibleInput.checked =
          controls.rotation;
      }

      if (
        typeof controls.graticule ===
        "boolean"
      ) {
        graticuleVisibleInput.checked =
          controls.graticule;
      }

      if (
        typeof controls.screenSpaceGraticule ===
        "boolean"
      ) {
        screenSpaceGraticuleVisibleInput.checked =
          controls.screenSpaceGraticule;
      }

      if (
        typeof controls.resetView ===
        "boolean"
      ) {
        resetViewVisibleInput.checked =
          controls.resetView;
      }

      if (
        typeof controls.fullscreen ===
        "boolean"
      ) {
        fullscreenVisibleInput.checked =
          controls.fullscreen;
      }

      if (
        typeof controls.mousePosition ===
        "boolean"
      ) {
        mousePositionVisibleInput.checked =
          controls.mousePosition;
      }

      if (
        typeof controls.overviewMap ===
        "boolean"
      ) {
        overviewMapVisibleInput.checked =
          controls.overviewMap;
      }
    }

    const navigation =
      savedSettings.navigation;

    if (
      navigation !== null &&
      typeof navigation === "object"
    ) {
      if (
        [
          "low",
          "default",
          "high",
        ].includes(
          navigation.mouseWheelZoomSensitivity,
        )
      ) {
        mouseWheelZoomSensitivitySelect.value =
          navigation.mouseWheelZoomSensitivity;
      }

      if (
        [
          "0.1",
          "0.5",
          "1",
          "2",
        ].includes(
          navigation.zoomButtonStep,
        )
      ) {
        zoomButtonStepSelect.value =
          navigation.zoomButtonStep;
      }
    }

    const overviewMapSettings =
      savedSettings.overviewMap;

    if (
      overviewMapSettings !== null &&
      typeof overviewMapSettings ===
        "object"
    ) {
      if (
        [
          "small",
          "default",
          "large",
        ].includes(
          overviewMapSettings.size,
        )
      ) {
        overviewMapSizeSelect.value =
          overviewMapSettings.size;
      }
    }

    const grid = savedSettings.grid;

    if (
      grid !== null &&
      typeof grid === "object"
    ) {
      if (
        [
          "default",
          "light",
          "dark",
          "light-contrast",
          "dark-contrast",
        ].includes(grid.theme)
      ) {
        gridThemeSelect.value =
          grid.theme;
      }

      const opacity =
        Number(grid.opacity);

      if (
        Number.isFinite(opacity) &&
        opacity >= 0 &&
        opacity <= 100
      ) {
        gridOpacityInput.value =
          opacity.toString();
      }

      if (
        [
          "fine",
          "default",
          "coarse",
        ].includes(grid.spacing)
      ) {
        gridSpacingSelect.value =
          grid.spacing;
      }

      if (
        typeof grid.labels ===
        "boolean"
      ) {
        gridLabelsVisibleInput.checked =
          grid.labels;
      }
    }

    const scaleBar =
      savedSettings.scaleBar;

    if (
      scaleBar !== null &&
      typeof scaleBar === "object"
    ) {
      if (
        typeof scaleBar.enabled ===
        "boolean"
      ) {
        scaleBarEnabledInput.checked =
          scaleBar.enabled;
      }

      if (
        typeof scaleBar.colour ===
          "string" &&
        /^#[0-9a-fA-F]{6}$/.test(
          scaleBar.colour,
        )
      ) {
        scaleBarColourInput.value =
          scaleBar.colour;
      }

      const opacity =
        Number(scaleBar.opacity);

      if (
        Number.isFinite(opacity) &&
        opacity >= 0 &&
        opacity <= 100
      ) {
        scaleBarOpacityInput.value =
          opacity.toString();
      }

      if (
        [
          "small",
          "default",
          "large",
        ].includes(scaleBar.size)
      ) {
        scaleBarSizeSelect.value =
          scaleBar.size;
      }

      if (
        [
          "metric",
          "imperial",
        ].includes(scaleBar.units)
      ) {
        scaleBarUnitsSelect.value =
          scaleBar.units;
      }
    }
  }

  function resetValues() {
    themeSelect.value = "dark";

    overviewMapSizeSelect.value =
      "default";

    mouseWheelZoomSensitivitySelect.value =
      "default";

    zoomButtonStepSelect.value = "1";

    gridThemeSelect.value = "default";
    gridOpacityInput.value = "50";
    gridSpacingSelect.value = "default";

    gridLabelsVisibleInput.checked =
      true;

    controlOpacityInput.value = "100";

    for (const input of [
      zoomVisibleInput,
      zoomLevelVisibleInput,
      rotationVisibleInput,
      graticuleVisibleInput,
      screenSpaceGraticuleVisibleInput,
      resetViewVisibleInput,
      fullscreenVisibleInput,
      mousePositionVisibleInput,
      overviewMapVisibleInput,
    ]) {
      input.checked = true;
    }

    scaleBarEnabledInput.checked = true;
    scaleBarColourInput.value =
      "#ffffff";

    scaleBarOpacityInput.value = "100";

    scaleBarSizeSelect.value =
      "default";

    scaleBarUnitsSelect.value =
      "metric";
  }

  function clearSavedSettings() {
    try {
      window.localStorage.removeItem(
        settingsStorageKey,
      );
    } catch {
      // The defaults still apply if storage is unavailable.
    }
  }

  function bindEvents() {
    if (eventsBound) {
      return;
    }

    eventsBound = true;

    toggle.addEventListener(
      "click",
      () => {
        const open =
          panel.classList.contains(
            "hidden",
          );

        setOpen(open);
      },
    );

    closeButton.addEventListener(
      "click",
      () => {
        setOpen(false);
      },
    );

    for (const tab of tabs) {
      tab.addEventListener(
        "click",
        () => {
          const selectedTab =
            tab.dataset.settingsTab;

          for (const otherTab of tabs) {
            otherTab.classList.toggle(
              "active",
              otherTab === tab,
            );
          }

          for (
            const tabPanel of
              tabPanels
          ) {
            tabPanel.classList.toggle(
              "hidden",
              tabPanel.dataset.settingsPanel !==
                selectedTab,
            );
          }
        },
      );
    }

    themeSelect.addEventListener(
      "change",
      () => {
        updateAppearance();
        onThemeChange();
        save();
      },
    );

    controlOpacityInput.addEventListener(
      "input",
      () => {
        updateAppearance();
        save();
      },
    );

    for (const input of [
      zoomVisibleInput,
      zoomLevelVisibleInput,
      rotationVisibleInput,
      graticuleVisibleInput,
      screenSpaceGraticuleVisibleInput,
      resetViewVisibleInput,
      fullscreenVisibleInput,
      mousePositionVisibleInput,
      overviewMapVisibleInput,
    ]) {
      input.addEventListener(
        "change",
        () => {
          onControlVisibilityChange();
          save();
        },
      );
    }

    for (const input of [
      gridThemeSelect,
      gridSpacingSelect,
      gridLabelsVisibleInput,
      overviewMapSizeSelect,
      mouseWheelZoomSensitivitySelect,
      zoomButtonStepSelect,
      scaleBarEnabledInput,
      scaleBarSizeSelect,
      scaleBarUnitsSelect,
    ]) {
      input.addEventListener(
        "change",
        () => {
          save();
        },
      );
    }

    for (const input of [
      gridOpacityInput,
      scaleBarColourInput,
      scaleBarOpacityInput,
    ]) {
      input.addEventListener(
        "input",
        () => {
          save();
        },
      );
    }

    resetDefaultsButton.addEventListener(
      "click",
      () => {
        onReset();
      },
    );
  }

  return {
    bindEvents,
    clearSavedSettings,
    load,
    resetValues,
    setOpen,
    updateAppearance,
  };
}

export { createSettingsPanelController };
