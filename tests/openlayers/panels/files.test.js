import {
  afterEach,
  beforeEach,
  describe,
  expect,
  it,
  vi,
} from "vitest";

import {
  createFilesPanelController,
} from "../../../tiatoolbox/visualization/openlayers/src/panels/files.js";

const configuredSlides = {
  directory: "/slides",
  files: [
    {
      name: "CMU-1.svs",
      path: "/slides/CMU-1.svs",
    },
    {
      name: "CMU-2.svs",
      path: "/slides/CMU-2.svs",
    },
  ],
};

const configuredOverlays = {
  directory: "/overlays",
  files: [
    {
      name: "CMU-1-mask.png",
      path: "/overlays/CMU-1-mask.png",
    },
    {
      name: "CMU-1-annotations.db",
      path: "/overlays/CMU-1-annotations.db",
    },
    {
      name: "CMU-2-mask.png",
      path: "/overlays/CMU-2-mask.png",
    },
  ],
};

function createDeferred() {
  let resolve;
  let reject;

  const promise = new Promise(
    (resolvePromise, rejectPromise) => {
      resolve = resolvePromise;
      reject = rejectPromise;
    },
  );

  return {
    promise,
    resolve,
    reject,
  };
}

async function flushAsyncEvents() {
  await new Promise((resolve) => {
    setTimeout(resolve, 0);
  });
}

function getButton(select) {
  return select.querySelector(
    ".viewer-file-select-button",
  );
}

function getLabel(select) {
  return select.querySelector(
    ".viewer-file-select-label",
  );
}

function getOptions(select) {
  return [
    ...select.querySelectorAll(
      ".viewer-file-select-option",
    ),
  ];
}

function createHarness({
  slidesConfig = configuredSlides,
  overlaysConfig = configuredOverlays,
  state: suppliedState,
  onSlideSelected,
  onOverlaySelected,
  onClearSlide,
  onClearOverlays,
  onOpen,
} = {}) {
  const state =
    suppliedState ?? {
      currentSlidePath: null,
      slideLoaded: false,
      overlaysLoaded: false,
    };

  const panel =
    document.createElement("div");
  panel.className = "hidden";

  const toggle =
    document.createElement("button");

  const container =
    document.createElement("div");

  panel.append(container);
  document.body.append(
    toggle,
    panel,
  );

  const slideSelected =
    onSlideSelected ??
    vi.fn(async (filePath) => {
      state.currentSlidePath =
        filePath;
      state.slideLoaded = true;
    });

  const overlaySelected =
    onOverlaySelected ??
    vi.fn(async () => {
      state.overlaysLoaded = true;
    });

  const clearSlide =
    onClearSlide ??
    vi.fn(async () => {
      state.currentSlidePath = null;
      state.slideLoaded = false;
      state.overlaysLoaded = false;
    });

  const clearOverlays =
    onClearOverlays ??
    vi.fn(async () => {
      state.overlaysLoaded = false;
    });

  const open =
    onOpen ?? vi.fn();

  const controller =
    createFilesPanelController({
      panel,
      toggle,
      container,
      configuredSlides:
        slidesConfig,
      configuredOverlays:
        overlaysConfig,
      getCurrentSlidePath: () =>
        state.currentSlidePath,
      hasSlide: () =>
        state.slideLoaded,
      hasOverlays: () =>
        state.overlaysLoaded,
      onSlideSelected:
        slideSelected,
      onOverlaySelected:
        overlaySelected,
      onClearSlide:
        clearSlide,
      onClearOverlays:
        clearOverlays,
      onOpen: open,
    });

  const selectors = [
    ...container.querySelectorAll(
      ".viewer-file-select",
    ),
  ];

  const [
    slideSelect,
    overlaySelect,
  ] = selectors;

  const actionButtons = [
    ...container.querySelectorAll(
      ".viewer-file-actions button",
    ),
  ];

  const [
    clearSlideButton,
    clearOverlaysButton,
  ] = actionButtons;

  return {
    state,
    panel,
    toggle,
    container,
    controller,
    slideSelect,
    overlaySelect,
    clearSlideButton,
    clearOverlaysButton,
    onSlideSelected:
      slideSelected,
    onOverlaySelected:
      overlaySelected,
    onClearSlide:
      clearSlide,
    onClearOverlays:
      clearOverlays,
    onOpen: open,
  };
}

beforeEach(() => {
  document.body.replaceChildren();

  vi.stubGlobal(
    "requestAnimationFrame",
    (callback) => {
      callback();
      return 1;
    },
  );
});

afterEach(() => {
  document.body.replaceChildren();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

describe("initialisation", () => {
  it("creates the file selectors and action buttons", () => {
    const {
      slideSelect,
      overlaySelect,
      clearSlideButton,
      clearOverlaysButton,
    } = createHarness();

    expect(
      getLabel(slideSelect)
        .textContent,
    ).toBe("Select slide");
    expect(
      getLabel(overlaySelect)
        .textContent,
    ).toBe("Select slide first");

    expect(
      slideSelect.disabled,
    ).toBe(false);
    expect(
      overlaySelect.disabled,
    ).toBe(true);

    expect(
      clearSlideButton.textContent,
    ).toBe("Clear Slide");
    expect(
      clearOverlaysButton.textContent,
    ).toBe("Clear Overlays");

    expect(
      clearSlideButton.disabled,
    ).toBe(true);
    expect(
      clearOverlaysButton.disabled,
    ).toBe(true);
  });

  it("shows placeholders when directories are not configured", () => {
    const {
      slideSelect,
      overlaySelect,
    } = createHarness({
      slidesConfig: {
        directory: null,
        files: [],
      },
      overlaysConfig: {
        directory: null,
        files: [],
      },
    });

    expect(
      getLabel(slideSelect)
        .textContent,
    ).toBe(
      "No slide directory configured",
    );
    expect(
      getLabel(overlaySelect)
        .textContent,
    ).toBe(
      "No overlay directory configured",
    );

    expect(
      slideSelect.disabled,
    ).toBe(true);
    expect(
      overlaySelect.disabled,
    ).toBe(true);
  });

  it("populates matching overlays for the current slide", () => {
    const state = {
      currentSlidePath:
        "/slides/CMU-1.svs",
      slideLoaded: true,
      overlaysLoaded: false,
    };

    const {
      overlaySelect,
    } = createHarness({
      state,
    });

    expect(
      overlaySelect.disabled,
    ).toBe(false);
    expect(
      getLabel(overlaySelect)
        .textContent,
    ).toBe("Load overlay");

    getButton(
      overlaySelect,
    ).click();

    expect(
      getOptions(
        overlaySelect,
      ).map(
        (option) =>
          option.textContent,
      ),
    ).toEqual([
      "CMU-1-mask.png",
      "CMU-1-annotations.db",
    ]);
  });

  it("shows when no overlays match the current slide", () => {
    const state = {
      currentSlidePath:
        "/slides/CMU-3.svs",
      slideLoaded: true,
      overlaysLoaded: false,
    };

    const {
      overlaySelect,
    } = createHarness({
      state,
    });

    expect(
      getLabel(overlaySelect)
        .textContent,
    ).toBe(
      "No matching overlays",
    );
    expect(
      overlaySelect.disabled,
    ).toBe(true);
  });
});

describe("controller state", () => {
  it("sets the selected slide", () => {
    const {
      controller,
      slideSelect,
    } = createHarness();

    controller.setSlide(
      "/slides/CMU-2.svs",
    );

    expect(
      slideSelect.value,
    ).toBe("/slides/CMU-2.svs");
    expect(
      getLabel(slideSelect)
        .textContent,
    ).toBe("CMU-2.svs");
  });

  it("updates the clear action state", () => {
    const {
      state,
      controller,
      clearSlideButton,
      clearOverlaysButton,
    } = createHarness();

    state.currentSlidePath =
      "/slides/CMU-1.svs";
    state.slideLoaded = true;

    controller.updateActionState();

    expect(
      clearSlideButton.disabled,
    ).toBe(false);
    expect(
      clearOverlaysButton.disabled,
    ).toBe(true);

    state.overlaysLoaded = true;

    controller.updateActionState();

    expect(
      clearSlideButton.disabled,
    ).toBe(false);
    expect(
      clearOverlaysButton.disabled,
    ).toBe(false);
  });

  it("opens and closes the panel", () => {
    const {
      panel,
      toggle,
      controller,
      slideSelect,
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
      toggle.innerHTML,
    ).toContain("fa-folder-open");
    expect(
      onOpen,
    ).toHaveBeenCalledOnce();

    getButton(
      slideSelect,
    ).click();

    expect(
      slideSelect.classList.contains(
        "open",
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
    expect(
      toggle.innerHTML,
    ).toContain("fa-folder");
    expect(
      toggle.innerHTML,
    ).not.toContain(
      "fa-folder-open",
    );
    expect(
      slideSelect.classList.contains(
        "open",
      ),
    ).toBe(false);
    expect(
      onOpen,
    ).toHaveBeenCalledOnce();
  });

  it("toggles the panel from the toggle button", () => {
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

describe("slide selection", () => {
  it("passes the selected slide to its callback", async () => {
    const {
      state,
      slideSelect,
      overlaySelect,
      clearSlideButton,
      onSlideSelected,
    } = createHarness();

    overlaySelect.value =
      "/overlays/old.db";

    slideSelect.dispatchEvent(
      new CustomEvent(
        "change",
        {
          detail:
            "/slides/CMU-1.svs",
        },
      ),
    );

    await flushAsyncEvents();

    expect(
      onSlideSelected,
    ).toHaveBeenCalledWith(
      "/slides/CMU-1.svs",
    );
    expect(
      state.slideLoaded,
    ).toBe(true);
    expect(
      overlaySelect.value,
    ).toBe("");
    expect(
      clearSlideButton.disabled,
    ).toBe(false);
  });

  it("disables all file controls while loading a slide", async () => {
    const deferred =
      createDeferred();

    const onSlideSelected =
      vi.fn(
        () => deferred.promise,
      );

    const {
      slideSelect,
      overlaySelect,
      clearSlideButton,
      clearOverlaysButton,
    } = createHarness({
      onSlideSelected,
    });

    slideSelect.dispatchEvent(
      new CustomEvent(
        "change",
        {
          detail:
            "/slides/CMU-1.svs",
        },
      ),
    );

    expect(
      slideSelect.disabled,
    ).toBe(true);
    expect(
      overlaySelect.disabled,
    ).toBe(true);
    expect(
      clearSlideButton.disabled,
    ).toBe(true);
    expect(
      clearOverlaysButton.disabled,
    ).toBe(true);

    deferred.resolve();

    await flushAsyncEvents();

    expect(
      slideSelect.disabled,
    ).toBe(false);
  });

  it("ignores an empty slide selection", async () => {
    const {
      slideSelect,
      onSlideSelected,
    } = createHarness();

    slideSelect.dispatchEvent(
      new CustomEvent(
        "change",
        {
          detail: "",
        },
      ),
    );

    await flushAsyncEvents();

    expect(
      onSlideSelected,
    ).not.toHaveBeenCalled();
  });

  it("uses the selector value when change detail is absent", async () => {
    const {
      slideSelect,
      onSlideSelected,
    } = createHarness();

    slideSelect.value =
      "/slides/CMU-2.svs";

    slideSelect.dispatchEvent(
      new CustomEvent("change"),
    );

    await flushAsyncEvents();

    expect(
      onSlideSelected,
    ).toHaveBeenCalledWith(
      "/slides/CMU-2.svs",
    );
  });

  it("logs slide loading errors and restores the controls", async () => {
    const error =
      new Error("Slide failure");

    const consoleError =
      vi.spyOn(
        console,
        "error",
      ).mockImplementation(
        () => {},
      );

    const {
      slideSelect,
    } = createHarness({
      onSlideSelected:
        vi.fn().mockRejectedValue(
          error,
        ),
    });

    slideSelect.dispatchEvent(
      new CustomEvent(
        "change",
        {
          detail:
            "/slides/CMU-1.svs",
        },
      ),
    );

    await flushAsyncEvents();

    expect(
      consoleError,
    ).toHaveBeenCalledWith(
      error,
    );
    expect(
      slideSelect.disabled,
    ).toBe(false);
  });
});

describe("overlay selection", () => {
  it("loads an overlay and clears the selection", async () => {
    const state = {
      currentSlidePath:
        "/slides/CMU-1.svs",
      slideLoaded: true,
      overlaysLoaded: false,
    };

    const {
      overlaySelect,
      clearOverlaysButton,
      onOverlaySelected,
    } = createHarness({
      state,
    });

    overlaySelect.value =
      "/overlays/CMU-1-mask.png";

    overlaySelect.dispatchEvent(
      new CustomEvent(
        "change",
        {
          detail:
            "/overlays/CMU-1-mask.png",
        },
      ),
    );

    await flushAsyncEvents();

    expect(
      onOverlaySelected,
    ).toHaveBeenCalledWith(
      "/overlays/CMU-1-mask.png",
    );
    expect(
      overlaySelect.value,
    ).toBe("");
    expect(
      clearOverlaysButton.disabled,
    ).toBe(false);
  });

  it("logs overlay loading errors without clearing the selection", async () => {
    const state = {
      currentSlidePath:
        "/slides/CMU-1.svs",
      slideLoaded: true,
      overlaysLoaded: false,
    };

    const error =
      new Error("Overlay failure");

    const consoleError =
      vi.spyOn(
        console,
        "error",
      ).mockImplementation(
        () => {},
      );

    const {
      overlaySelect,
    } = createHarness({
      state,
      onOverlaySelected:
        vi.fn().mockRejectedValue(
          error,
        ),
    });

    overlaySelect.value =
      "/overlays/CMU-1-mask.png";

    overlaySelect.dispatchEvent(
      new CustomEvent("change"),
    );

    await flushAsyncEvents();

    expect(
      consoleError,
    ).toHaveBeenCalledWith(
      error,
    );
    expect(
      overlaySelect.value,
    ).toBe(
      "/overlays/CMU-1-mask.png",
    );
  });
});

describe("clear actions", () => {
  it("clears the slide and resets both selectors", async () => {
    const state = {
      currentSlidePath:
        "/slides/CMU-1.svs",
      slideLoaded: true,
      overlaysLoaded: true,
    };

    const {
      controller,
      slideSelect,
      overlaySelect,
      clearSlideButton,
      clearOverlaysButton,
      onClearSlide,
    } = createHarness({
      state,
    });

    controller.setSlide(
      "/slides/CMU-1.svs",
    );
    overlaySelect.value =
      "/overlays/CMU-1-mask.png";

    clearSlideButton.click();

    await flushAsyncEvents();

    expect(
      onClearSlide,
    ).toHaveBeenCalledOnce();
    expect(
      slideSelect.value,
    ).toBe("");
    expect(
      overlaySelect.value,
    ).toBe("");
    expect(
      getLabel(overlaySelect)
        .textContent,
    ).toBe("Select slide first");
    expect(
      clearSlideButton.disabled,
    ).toBe(true);
    expect(
      clearOverlaysButton.disabled,
    ).toBe(true);
  });

  it("preserves selections when clearing the slide fails", async () => {
    const state = {
      currentSlidePath:
        "/slides/CMU-1.svs",
      slideLoaded: true,
      overlaysLoaded: true,
    };

    const error =
      new Error("Clear slide failure");

    const consoleError =
      vi.spyOn(
        console,
        "error",
      ).mockImplementation(
        () => {},
      );

    const {
      controller,
      slideSelect,
      overlaySelect,
      clearSlideButton,
    } = createHarness({
      state,
      onClearSlide:
        vi.fn().mockRejectedValue(
          error,
        ),
    });

    controller.setSlide(
      "/slides/CMU-1.svs",
    );
    overlaySelect.value =
      "/overlays/CMU-1-mask.png";

    clearSlideButton.click();

    await flushAsyncEvents();

    expect(
      consoleError,
    ).toHaveBeenCalledWith(
      error,
    );
    expect(
      slideSelect.value,
    ).toBe("/slides/CMU-1.svs");
    expect(
      overlaySelect.value,
    ).toBe(
      "/overlays/CMU-1-mask.png",
    );
  });

  it("clears overlays and resets the overlay selector", async () => {
    const state = {
      currentSlidePath:
        "/slides/CMU-1.svs",
      slideLoaded: true,
      overlaysLoaded: true,
    };

    const {
      overlaySelect,
      clearOverlaysButton,
      onClearOverlays,
    } = createHarness({
      state,
    });

    overlaySelect.value =
      "/overlays/CMU-1-mask.png";

    clearOverlaysButton.click();

    await flushAsyncEvents();

    expect(
      onClearOverlays,
    ).toHaveBeenCalledOnce();
    expect(
      overlaySelect.value,
    ).toBe("");
    expect(
      clearOverlaysButton.disabled,
    ).toBe(true);
  });

  it("preserves the overlay selection when clearing overlays fails", async () => {
    const state = {
      currentSlidePath:
        "/slides/CMU-1.svs",
      slideLoaded: true,
      overlaysLoaded: true,
    };

    const error =
      new Error(
        "Clear overlays failure",
      );

    const consoleError =
      vi.spyOn(
        console,
        "error",
      ).mockImplementation(
        () => {},
      );

    const {
      overlaySelect,
      clearOverlaysButton,
    } = createHarness({
      state,
      onClearOverlays:
        vi.fn().mockRejectedValue(
          error,
        ),
    });

    overlaySelect.value =
      "/overlays/CMU-1-mask.png";

    clearOverlaysButton.click();

    await flushAsyncEvents();

    expect(
      consoleError,
    ).toHaveBeenCalledWith(
      error,
    );
    expect(
      overlaySelect.value,
    ).toBe(
      "/overlays/CMU-1-mask.png",
    );
  });
});
