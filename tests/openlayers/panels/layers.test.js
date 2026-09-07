import {
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

import {
  createLayersPanelController,
} from "../../../tiatoolbox/visualization/openlayers/src/panels/layers.js";

function createLayer({
  source = {},
  visible = true,
  opacity = 1,
  zIndex = 0,
} = {}) {
  let currentVisible = visible;
  let currentOpacity = opacity;
  let currentZIndex = zIndex;

  return {
    getSource: vi.fn(
      () => source,
    ),

    getVisible: vi.fn(
      () => currentVisible,
    ),

    setVisible: vi.fn(
      (newVisible) => {
        currentVisible =
          newVisible;
      },
    ),

    getOpacity: vi.fn(
      () => currentOpacity,
    ),

    setOpacity: vi.fn(
      (newOpacity) => {
        currentOpacity =
          newOpacity;
      },
    ),

    getZIndex: vi.fn(
      () => currentZIndex,
    ),

    setZIndex: vi.fn(
      (newZIndex) => {
        currentZIndex =
          newZIndex;
      },
    ),
  };
}

function createHarness({
  slideLayer =
    createLayer(),
  currentSlidePath =
    "/slides/CMU-1.svs",
  overlayLayers = {},
  onRemoveLayer,
  onOpen,
} = {}) {
  const panel =
    document.createElement(
      "div",
    );

  panel.className = "hidden";

  const toggle =
    document.createElement(
      "button",
    );

  const list =
    document.createElement(
      "div",
    );

  panel.append(list);

  document.body.append(
    toggle,
    panel,
  );

  const removeLayer =
    onRemoveLayer ??
    vi.fn().mockResolvedValue(
      undefined,
    );

  const open =
    onOpen ?? vi.fn();

  const controller =
    createLayersPanelController({
      panel,
      toggle,
      list,
      getSlideLayer: () =>
        slideLayer,
      getCurrentSlidePath: () =>
        currentSlidePath,
      getOverlayLayers: () =>
        overlayLayers,
      onRemoveLayer:
        removeLayer,
      onOpen: open,
    });

  return {
    panel,
    toggle,
    list,
    controller,
    slideLayer,
    overlayLayers,
    onRemoveLayer:
      removeLayer,
    onOpen: open,
  };
}

function getItems(list) {
  return [
    ...list.querySelectorAll(
      ".layer-editor-item",
    ),
  ];
}

function getItemName(item) {
  return item.querySelector(
    ".layer-editor-name",
  ).textContent;
}

function getItemByName(
  list,
  layerName,
) {
  return getItems(list).find(
    (item) =>
      getItemName(item) ===
      layerName,
  );
}

async function flushPromises() {
  await Promise.resolve();
  await Promise.resolve();
}

beforeEach(() => {
  document.body.replaceChildren();
});

describe("rendering", () => {
  it("shows an empty state when no layers are loaded", () => {
    const {
      list,
      controller,
    } = createHarness({
      slideLayer:
        createLayer({
          source: null,
        }),
    });

    controller.render();

    expect(
      getItems(list),
    ).toHaveLength(0);

    expect(
      list.querySelector(
        ".layer-editor-empty",
      ).textContent,
    ).toBe("No layers loaded");
  });

  it("renders the slide first and overlays in z-index order", () => {
    const lowerOverlay =
      createLayer({
        zIndex: 10,
      });

    const upperOverlay =
      createLayer({
        zIndex: 20,
      });

    const {
      list,
      controller,
    } = createHarness({
      overlayLayers: {
        Upper: upperOverlay,
        Lower: lowerOverlay,
      },
    });

    controller.render();

    expect(
      getItems(list).map(
        getItemName,
      ),
    ).toEqual([
      "CMU-1",
      "Lower",
      "Upper",
    ]);
  });

  it("uses a fallback name when the slide path is unavailable", () => {
    const {
      list,
      controller,
    } = createHarness({
      currentSlidePath: null,
    });

    controller.render();

    expect(
      getItemName(
        getItems(list)[0],
      ),
    ).toBe("slide");
  });

  it("only gives overlays ordering and removal controls", () => {
    const first =
      createLayer({
        zIndex: 10,
      });

    const second =
      createLayer({
        zIndex: 20,
      });

    const {
      list,
      controller,
    } = createHarness({
      overlayLayers: {
        First: first,
        Second: second,
      },
    });

    controller.render();

    const slideItem =
      getItemByName(
        list,
        "CMU-1",
      );

    expect(
      slideItem.querySelector(
        ".layer-editor-order",
      ),
    ).toBeNull();

    const firstItem =
      getItemByName(
        list,
        "First",
      );

    const secondItem =
      getItemByName(
        list,
        "Second",
      );

    const firstButtons = [
      ...firstItem.querySelectorAll(
        ".layer-editor-order button",
      ),
    ];

    const secondButtons = [
      ...secondItem.querySelectorAll(
        ".layer-editor-order button",
      ),
    ];

    expect(firstButtons).toHaveLength(
      3,
    );
    expect(secondButtons).toHaveLength(
      3,
    );

    expect(
      firstButtons[0].title,
    ).toBe("Move layer up");
    expect(
      firstButtons[0].disabled,
    ).toBe(true);

    expect(
      firstButtons[1].disabled,
    ).toBe(false);

    expect(
      secondButtons[0].disabled,
    ).toBe(false);

    expect(
      secondButtons[1].disabled,
    ).toBe(true);

    expect(
      secondButtons[2].title,
    ).toBe("Remove Second");
  });
});

describe("layer controls", () => {
  it("updates layer visibility", () => {
    const overlay =
      createLayer({
        visible: true,
      });

    const {
      list,
      controller,
    } = createHarness({
      overlayLayers: {
        Overlay: overlay,
      },
    });

    controller.render();

    const item =
      getItemByName(
        list,
        "Overlay",
      );

    const visibility =
      item.querySelector(
        ".layer-editor-visibility",
      );

    expect(
      visibility.checked,
    ).toBe(true);

    visibility.checked = false;

    visibility.dispatchEvent(
      new Event(
        "change",
        {
          bubbles: true,
        },
      ),
    );

    expect(
      overlay.setVisible,
    ).toHaveBeenCalledWith(
      false,
    );

    expect(
      overlay.getVisible(),
    ).toBe(false);
  });

  it("updates layer opacity and its percentage label", () => {
    const overlay =
      createLayer({
        opacity: 0.75,
      });

    const {
      list,
      controller,
    } = createHarness({
      overlayLayers: {
        Overlay: overlay,
      },
    });

    controller.render();

    const item =
      getItemByName(
        list,
        "Overlay",
      );

    const slider =
      item.querySelector(
        ".layer-editor-slider",
      );

    const value =
      item.querySelector(
        ".layer-editor-value",
      );

    expect(slider.value).toBe(
      "0.75",
    );
    expect(value.textContent).toBe(
      "75%",
    );

    slider.value = "0.35";

    slider.dispatchEvent(
      new Event(
        "input",
        {
          bubbles: true,
        },
      ),
    );

    expect(
      overlay.setOpacity,
    ).toHaveBeenCalledWith(
      0.35,
    );

    expect(
      overlay.getOpacity(),
    ).toBe(0.35);

    expect(value.textContent).toBe(
      "35%",
    );
  });

  it("moves an overlay down by swapping z-index values", () => {
    const first =
      createLayer({
        zIndex: 10,
      });

    const second =
      createLayer({
        zIndex: 20,
      });

    const {
      list,
      controller,
    } = createHarness({
      overlayLayers: {
        First: first,
        Second: second,
      },
    });

    controller.render();

    const firstItem =
      getItemByName(
        list,
        "First",
      );

    const moveDown =
      firstItem.querySelector(
        'button[title="Move layer down"]',
      );

    moveDown.click();

    expect(
      first.getZIndex(),
    ).toBe(20);

    expect(
      second.getZIndex(),
    ).toBe(10);

    expect(
      getItems(list).map(
        getItemName,
      ),
    ).toEqual([
      "CMU-1",
      "Second",
      "First",
    ]);
  });

  it("moves an overlay up by swapping z-index values", () => {
    const first =
      createLayer({
        zIndex: 10,
      });

    const second =
      createLayer({
        zIndex: 20,
      });

    const {
      list,
      controller,
    } = createHarness({
      overlayLayers: {
        First: first,
        Second: second,
      },
    });

    controller.render();

    const secondItem =
      getItemByName(
        list,
        "Second",
      );

    const moveUp =
      secondItem.querySelector(
        'button[title="Move layer up"]',
      );

    moveUp.click();

    expect(
      first.getZIndex(),
    ).toBe(20);

    expect(
      second.getZIndex(),
    ).toBe(10);

    expect(
      getItems(list).map(
        getItemName,
      ),
    ).toEqual([
      "CMU-1",
      "Second",
      "First",
    ]);
  });

  it("passes an overlay ID to the remove callback", async () => {
    const overlay =
      createLayer();

    const {
      list,
      controller,
      onRemoveLayer,
    } = createHarness({
      overlayLayers: {
        Tumour: overlay,
      },
    });

    controller.render();

    const item =
      getItemByName(
        list,
        "Tumour",
      );

    item.querySelector(
      'button[title="Remove Tumour"]',
    ).click();

    await flushPromises();

    expect(
      onRemoveLayer,
    ).toHaveBeenCalledOnce();

    expect(
      onRemoveLayer,
    ).toHaveBeenCalledWith(
      "Tumour",
    );
  });

  it("logs errors when removing an overlay fails", async () => {
    const error =
      new Error(
        "Remove failure",
      );

    const consoleError =
      vi.spyOn(
        console,
        "error",
      ).mockImplementation(
        () => {},
      );

    const {
      list,
      controller,
    } = createHarness({
      overlayLayers: {
        Tumour:
          createLayer(),
      },

      onRemoveLayer:
        vi.fn().mockRejectedValue(
          error,
        ),
    });

    controller.render();

    getItemByName(
      list,
      "Tumour",
    )
      .querySelector(
        'button[title="Remove Tumour"]',
      )
      .click();

    await flushPromises();

    expect(
      consoleError,
    ).toHaveBeenCalledWith(
      error,
    );
  });
});

describe("panel state", () => {
  it("opens and closes through the controller", () => {
    const {
      panel,
      toggle,
      controller,
      onOpen,
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

    expect(
      onOpen,
    ).toHaveBeenCalledOnce();

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

    expect(
      onOpen,
    ).toHaveBeenCalledOnce();
  });

  it("toggles the panel from its toggle button", () => {
    const {
      panel,
      toggle,
      onOpen,
    } = createHarness();

    toggle.click();

    expect(
      panel.classList.contains(
        "hidden",
      ),
    ).toBe(false);

    expect(
      onOpen,
    ).toHaveBeenCalledOnce();

    toggle.click();

    expect(
      panel.classList.contains(
        "hidden",
      ),
    ).toBe(true);

    expect(
      onOpen,
    ).toHaveBeenCalledOnce();
  });
});
