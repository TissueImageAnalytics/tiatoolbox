import { createFileSelect } from "../components/file-select.js";
import { getFileStem } from "../utils/paths.js";

function createFilesPanelController({
  panel,
  toggle,
  container,
  configuredSlides,
  configuredOverlays,
  getCurrentSlidePath,
  hasSlide,
  hasOverlays,
  onSlideSelected,
  onOverlaySelected,
  onClearSlide,
  onClearOverlays,
  onOpen,
}) {
  const fileSelectors =
    document.createElement("div");

  fileSelectors.className =
    "viewer-file-selectors";

  const slideSelect =
    createFileSelect("Select slide");

  const overlaySelect =
    createFileSelect("Load overlay");

  fileSelectors.append(
    slideSelect,
    overlaySelect,
  );

  const fileActions =
    document.createElement("div");

  fileActions.className =
    "viewer-file-actions";

  function createFileActionButton(label) {
    const button =
      document.createElement("button");

    button.type = "button";
    button.textContent = label;

    return button;
  }

  const clearSlideButton =
    createFileActionButton("Clear Slide");

  const clearOverlaysButton =
    createFileActionButton("Clear Overlays");

  clearSlideButton.disabled = true;
  clearOverlaysButton.disabled = true;

  fileActions.append(
    clearSlideButton,
    clearOverlaysButton,
  );

  container.append(
    fileSelectors,
    fileActions,
  );

  function setOpen(open) {
    if (open) {
      onOpen();
    }

    panel.classList.toggle(
      "hidden",
      !open,
    );

    toggle.classList.toggle(
      "active",
      open,
    );

    toggle.innerHTML = open
      ? '<i class="fas fa-folder-open"></i>'
      : '<i class="fas fa-folder"></i>';

    if (!open) {
      for (
        const select of
          panel.querySelectorAll(
            ".viewer-file-select.open",
          )
      ) {
        select.close?.();
      }
    }
  }

  function populateFileSelect(
    select,
    files,
    placeholder,
  ) {
    select.setFiles(
      files,
      placeholder,
    );
  }

  function getMatchingOverlays(slidePath) {
    const slideStem =
      getFileStem(slidePath);

    return configuredOverlays.files.filter(
      (file) => {
        const fileName =
          file.name
            .split(/[\\/]/)
            .pop() ?? file.name;

        return fileName.includes(
          slideStem,
        );
      },
    );
  }

  function updateOverlaySelect() {
    if (
      configuredOverlays.directory === null
    ) {
      populateFileSelect(
        overlaySelect,
        [],
        "No overlay directory configured",
      );

      return;
    }

    const currentSlidePath =
      getCurrentSlidePath();

    if (currentSlidePath === null) {
      populateFileSelect(
        overlaySelect,
        [],
        "Select slide first",
      );

      return;
    }

    const matchingOverlays =
      getMatchingOverlays(
        currentSlidePath,
      );

    populateFileSelect(
      overlaySelect,
      matchingOverlays,
      matchingOverlays.length === 0
        ? "No matching overlays"
        : "Load overlay",
    );
  }

  function setSlide(filePath) {
    slideSelect.value = filePath;
  }

  function updateActionState() {
    const slideLoaded = hasSlide();

    const overlaysLoaded =
      hasOverlays();

    const currentSlidePath =
      getCurrentSlidePath();

    const hasMatchingOverlays =
      currentSlidePath !== null &&
      getMatchingOverlays(
        currentSlidePath,
      ).length > 0;

    clearSlideButton.disabled =
      !slideLoaded;

    clearOverlaysButton.disabled =
      !slideLoaded ||
      !overlaysLoaded;

    slideSelect.disabled =
      configuredSlides.files.length === 0;

    overlaySelect.disabled =
      !slideLoaded ||
      !hasMatchingOverlays;
  }

  function setBusy(busy) {
    if (!busy) {
      updateActionState();
      return;
    }

    slideSelect.disabled = true;
    overlaySelect.disabled = true;
    clearSlideButton.disabled = true;
    clearOverlaysButton.disabled = true;
  }

  slideSelect.addEventListener(
    "change",
    async (event) => {
      const filePath =
        event.detail ??
        slideSelect.value;

      if (filePath === "") {
        return;
      }

      setBusy(true);

      try {
        await onSlideSelected(
          filePath,
        );

        overlaySelect.value = "";
      } catch (error) {
        console.error(error);
      } finally {
        setBusy(false);
      }
    },
  );

  overlaySelect.addEventListener(
    "change",
    async (event) => {
      const filePath =
        event.detail ??
        overlaySelect.value;

      if (filePath === "") {
        return;
      }

      setBusy(true);

      try {
        await onOverlaySelected(
          filePath,
        );

        overlaySelect.value = "";
      } catch (error) {
        console.error(error);
      } finally {
        setBusy(false);
      }
    },
  );

  clearSlideButton.addEventListener(
    "click",
    async () => {
      setBusy(true);

      try {
        await onClearSlide();

        slideSelect.value = "";
        overlaySelect.value = "";

        updateOverlaySelect();
      } catch (error) {
        console.error(error);
      } finally {
        setBusy(false);
      }
    },
  );

  clearOverlaysButton.addEventListener(
    "click",
    async () => {
      setBusy(true);

      try {
        await onClearOverlays();

        overlaySelect.value = "";
      } catch (error) {
        console.error(error);
      } finally {
        setBusy(false);
      }
    },
  );

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

  populateFileSelect(
    slideSelect,
    configuredSlides.files,
    configuredSlides.directory === null
      ? "No slide directory configured"
      : "Select slide",
  );

  updateOverlaySelect();
  updateActionState();

  return {
    setOpen,
    setSlide,
    updateActionState,
    updateOverlaySelect,
  };
}

export { createFilesPanelController };
