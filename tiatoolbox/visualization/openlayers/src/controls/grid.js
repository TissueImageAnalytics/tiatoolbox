import Graticule from "ol-ext/control/Graticule.js";
import Toggle from "ol-ext/control/Toggle.js";
import Fill from "ol/style/Fill.js";
import Stroke from "ol/style/Stroke.js";
import Style from "ol/style/Style.js";
import Text from "ol/style/Text.js";

import { toRgba } from "../utils/colours.js";

const gridSpacingValues = {
  fine: 32,
  default: 64,
  coarse: 128,
};

const graticuleMargin = 64;

const gridThemeColours = {
  light: {
    line: { r: 255, g: 255, b: 255 },
    label: "rgba(255, 255, 255, 1)",
    outline: "rgba(20, 20, 20, 1)",
  },

  dark: {
    line: { r: 20, g: 20, b: 20 },
    label: "rgba(20, 20, 20, 1)",
    outline: "rgba(255, 255, 255, 1)",
  },

  "light-contrast": {
    line: { r: 0, g: 170, b: 200 },
    label: "rgba(0, 170, 200, 1)",
    outline: "rgba(20, 20, 20, 1)",
  },

  "dark-contrast": {
    line: { r: 145, g: 55, b: 0 },
    label: "rgba(145, 55, 0, 1)",
    outline: "rgba(255, 255, 255, 1)",
  },
};

function createGridController({
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
}) {
  function getGridSpacing() {
    return (
      gridSpacingValues[gridSpacingSelect.value] ??
      gridSpacingValues.default
    );
  }

  function getGridTheme() {
    if (gridThemeSelect.value !== "default") {
      return gridThemeSelect.value;
    }

    if (themeSelect.value === "light") {
      return "light";
    }

    if (themeSelect.value === "high-contrast") {
      return "dark-contrast";
    }

    return "dark";
  }

  const graticuleTextStyle = new Text({
    font: "12px Calibri,sans-serif",
    fill: new Fill({
      color: "rgba(0, 0, 0, 1)",
    }),
    stroke: new Stroke({
      color: "rgba(255, 255, 255, 1)",
      width: 3,
    }),
  });

  const graticuleStyle = new Style({
    stroke: new Stroke({
      color: "rgba(0, 0, 0, 0.5)",
      width: 1,
    }),
    text: graticuleTextStyle,
  });

  function updateAppearance() {
    const gridTheme = getGridTheme();

    const colours =
      gridThemeColours[gridTheme] ??
      gridThemeColours.dark;

    const opacity =
      Number(gridOpacityInput.value) / 100;

    const gridStroke = graticuleStyle.getStroke();
    const gridText = graticuleTextStyle;

    gridStroke.setColor(
      toRgba(colours.line, opacity),
    );

    gridText
      .getFill()
      .setColor(colours.label);

    gridText
      .getStroke()
      .setColor(colours.outline);

    gridOpacityValue.textContent =
      `${gridOpacityInput.value}%`;

    map.renderSync();
  }

  function updateLabels() {
    graticuleStyle.setText(
      gridLabelsVisibleInput.checked
        ? graticuleTextStyle
        : null,
    );

    map.renderSync();
  }

  function createGraticule(graticuleProjection) {
    return new Graticule({
      projection: graticuleProjection,
      margin: graticuleMargin,
      style: graticuleStyle,
      spacing: getGridSpacing(),
      formatCoord: (coordinate, position) => {
        if (
          position === "left" ||
          position === "right"
        ) {
          coordinate = -Math.floor(coordinate);
        } else {
          coordinate = Math.floor(coordinate);
        }

        if (coordinate >= 1e6) {
          coordinate =
            coordinate.toExponential(3);

          coordinate =
            coordinate.replace("+", "");
        }

        return coordinate;
      },
    });
  }

  const screenSpaceGraticuleMargin =
    graticuleMargin;

  function createScreenSpaceGraticule(
    graticuleProjection,
  ) {
    const spacing = getGridSpacing();

    return new Graticule({
      projection: graticuleProjection.getCode(),
      spacing,
      margin: screenSpaceGraticuleMargin,
      style: graticuleStyle,

      formatCoord(coordinate, position) {
        const mapExtent = map
          .getView()
          .calculateExtent(map.getSize());

        const resolution =
          map.getView().getResolution();

        const xOrigin =
          mapExtent[0] +
          resolution * screenSpaceGraticuleMargin;

        const yOrigin =
          mapExtent[3] -
          resolution * screenSpaceGraticuleMargin;

        let displayedCoordinate;

        if (
          position === "left" ||
          position === "right"
        ) {
          displayedCoordinate =
            -(coordinate - yOrigin);
        } else {
          displayedCoordinate =
            coordinate - xOrigin;
        }

        displayedCoordinate = Math.floor(
          displayedCoordinate /
            resolution /
            spacing,
        );

        if (
          position === "left" ||
          position === "right"
        ) {
          let string = "";

          do {
            string += String.fromCharCode(
              65 + (displayedCoordinate % 26),
            );

            displayedCoordinate = Math.floor(
              displayedCoordinate / 26,
            );
          } while (displayedCoordinate > 0);

          return string
            .split("")
            .reverse()
            .join("");
        }

        return displayedCoordinate;
      },
    });
  }

  let graticule =
    createGraticule(projection);

  let screenSpaceGraticule =
    createScreenSpaceGraticule(projection);

  function notifyGraticulesChange() {
    onGraticulesChange(
      graticule,
      screenSpaceGraticule,
    );
  }

  const graticuleToggle = new Toggle({
    html: '<i class="fas fa-ruler-combined"></i>',
    className: "ol-graticule",
    title: "Toggle Graticule",

    onToggle(active) {
      graticuleToggle.element.classList.toggle(
        "active",
        active,
      );

      if (active) {
        screenSpaceGraticuleToggle.setActive(
          false,
        );

        screenSpaceGraticuleToggle.element.classList.remove(
          "active",
        );

        screenSpaceGraticule.setMap(null);
        graticule.setMap(map);
      } else {
        graticule.setMap(null);
      }
    },
  });

  map.addControl(graticuleToggle);

  const screenSpaceGraticuleToggle =
    new Toggle({
      html: '<i class="fas fa-border-all"></i>',
      className:
        "ol-screen-space-graticule",
      title: "Toggle Screen Space Graticule",

      onToggle(active) {
        screenSpaceGraticuleToggle.element.classList.toggle(
          "active",
          active,
        );

        if (active) {
          graticuleToggle.setActive(false);

          graticuleToggle.element.classList.remove(
            "active",
          );

          graticule.setMap(null);
          screenSpaceGraticule.setMap(map);
        } else {
          screenSpaceGraticule.setMap(null);
        }
      },
    });

  map.addControl(screenSpaceGraticuleToggle);

  function recreateGraticules(
    graticuleProjection,
    preserveActive,
  ) {
    let graticuleWasActive = false;
    let screenSpaceGraticuleWasActive = false;

    if (preserveActive) {
      graticuleWasActive =
        graticuleToggle.getActive();

      screenSpaceGraticuleWasActive =
        screenSpaceGraticuleToggle.getActive();
    } else {
      graticuleToggle.setActive(false);
      screenSpaceGraticuleToggle.setActive(false);

      graticuleToggle.element.classList.remove(
        "active",
      );

      screenSpaceGraticuleToggle.element.classList.remove(
        "active",
      );
    }

    graticule.setMap(null);
    screenSpaceGraticule.setMap(null);

    graticule =
      createGraticule(graticuleProjection);

    screenSpaceGraticule =
      createScreenSpaceGraticule(
        graticuleProjection,
      );

    if (graticuleWasActive) {
      graticule.setMap(map);
    }

    if (screenSpaceGraticuleWasActive) {
      screenSpaceGraticule.setMap(map);
    }

    notifyGraticulesChange();
    map.renderSync();
  }

  function updateSpacing() {
    recreateGraticules(
      map.getView().getProjection(),
      true,
    );
  }

  function setProjection(
    graticuleProjection,
    { preserveActive = true } = {},
  ) {
    recreateGraticules(
      graticuleProjection,
      preserveActive,
    );
  }

  function updateVisibility() {
    graticuleToggle.element.classList.toggle(
      "viewer-control-hidden",
      !graticuleVisibleInput.checked,
    );

    screenSpaceGraticuleToggle.element.classList.toggle(
      "viewer-control-hidden",
      !screenSpaceGraticuleVisibleInput.checked,
    );

    if (!graticuleVisibleInput.checked) {
      graticuleToggle.setActive(false);

      graticuleToggle.element.classList.remove(
        "active",
      );

      graticule.setMap(null);
    }

    if (
      !screenSpaceGraticuleVisibleInput.checked
    ) {
      screenSpaceGraticuleToggle.setActive(false);

      screenSpaceGraticuleToggle.element.classList.remove(
        "active",
      );

      screenSpaceGraticule.setMap(null);
    }
  }

  function setViewerEnabled(enabled) {
    const graticuleButton =
      graticuleToggle.element.querySelector(
        "button",
      );

    const screenSpaceGraticuleButton =
      screenSpaceGraticuleToggle.element.querySelector(
        "button",
      );

    for (const button of [
      graticuleButton,
      screenSpaceGraticuleButton,
    ]) {
      if (button !== null) {
        button.disabled = !enabled;
      }
    }

    if (!enabled) {
      graticuleToggle.setActive(false);
      screenSpaceGraticuleToggle.setActive(false);

      graticuleToggle.element.classList.remove(
        "active",
      );

      screenSpaceGraticuleToggle.element.classList.remove(
        "active",
      );

      graticule.setMap(null);
      screenSpaceGraticule.setMap(null);
    }
  }

  gridThemeSelect.addEventListener(
    "change",
    () => {
      updateAppearance();
    },
  );

  gridOpacityInput.addEventListener(
    "input",
    () => {
      updateAppearance();
    },
  );

  gridLabelsVisibleInput.addEventListener(
    "change",
    () => {
      updateLabels();
    },
  );

  gridSpacingSelect.addEventListener(
    "change",
    () => {
      updateSpacing();
    },
  );

  updateAppearance();
  updateLabels();
  notifyGraticulesChange();

  return {
    graticuleToggle,
    screenSpaceGraticuleToggle,
    setProjection,
    setViewerEnabled,
    updateAppearance,
    updateLabels,
    updateSpacing,
    updateVisibility,
  };
}

export { createGridController };
