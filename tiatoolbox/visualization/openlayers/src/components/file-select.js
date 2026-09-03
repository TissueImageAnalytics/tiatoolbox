let fileSelectId = 0;

function createFileSelect(placeholder) {
  const select = document.createElement("div");
  select.className = "viewer-file-select";

  const button = document.createElement("button");
  button.type = "button";
  button.className = "viewer-file-select-button";
  button.setAttribute("aria-haspopup", "listbox");
  button.setAttribute("aria-expanded", "false");

  const label = document.createElement("span");
  label.className = "viewer-file-select-label";
  label.textContent = placeholder;

  button.append(label);

  const menu = document.createElement("div");
  menu.className = "viewer-file-select-menu";
  menu.hidden = true;

  const search = document.createElement("input");
  search.type = "text";
  search.className = "viewer-file-select-search";
  search.placeholder = "Search";
  search.autocomplete = "off";
  search.spellcheck = false;
  search.setAttribute(
    "aria-label",
    `Search ${placeholder.toLowerCase()}`,
  );

  const options = document.createElement("div");
  options.className = "viewer-file-select-options";
  options.id = `viewer-file-select-${fileSelectId}`;
  options.setAttribute("role", "listbox");

  fileSelectId += 1;

  button.setAttribute("aria-controls", options.id);
  search.setAttribute("aria-controls", options.id);

  menu.append(search, options);
  select.append(button, menu);

  let files = [];
  let selectedPath = "";
  let currentPlaceholder = placeholder;
  let filteredFiles = [];
  let activeIndex = -1;

  function setExpanded(expanded) {
    select.classList.toggle("open", expanded);
    menu.hidden = !expanded;
    button.setAttribute(
      "aria-expanded",
      expanded.toString(),
    );
  }

  function updateLabel() {
    if (selectedPath === "") {
      label.textContent = currentPlaceholder;
      label.title = "";
      return;
    }

    const selectedFile = files.find(
      (file) => file.path === selectedPath,
    );

    const fileName =
      selectedFile?.name ??
      selectedPath.split(/[\\/]/).pop() ??
      selectedPath;

    label.textContent = fileName;
    label.title = selectedPath;
  }

  function selectFile(file) {
    selectedPath = file.path;
    updateLabel();
    close();

    select.dispatchEvent(
      new CustomEvent("change", {
        detail: file.path,
      }),
    );
  }

  function renderOptions() {
    const query = search.value
      .trim()
      .toLocaleLowerCase();

    filteredFiles = files.filter(
      (file) =>
        file.name
          .toLocaleLowerCase()
          .includes(query),
    );

    options.replaceChildren();

    if (filteredFiles.length === 0) {
      const empty = document.createElement("div");
      empty.className = "viewer-file-select-empty";
      empty.textContent = "No matches";
      options.append(empty);
      return;
    }

    filteredFiles.forEach((file, index) => {
      const option = document.createElement("button");

      option.type = "button";
      option.className =
        "viewer-file-select-option";
      option.textContent = file.name;
      option.title = file.path;
      option.setAttribute("role", "option");
      option.setAttribute(
        "aria-selected",
        (file.path === selectedPath).toString(),
      );

      if (file.path === selectedPath) {
        option.classList.add("selected");
      }

      if (index === activeIndex) {
        option.classList.add("active");
      }

      option.addEventListener("mousedown", (event) => {
        event.preventDefault();
      });

      option.addEventListener("click", (event) => {
        event.stopPropagation();
        selectFile(file);
      });

      options.append(option);
    });

    options
      .querySelector(
        ".viewer-file-select-option.active",
      )
      ?.scrollIntoView({
        block: "nearest",
      });
  }

  function open() {
    if (button.disabled || files.length === 0) {
      return;
    }

    for (const otherSelect of document.querySelectorAll(
      ".viewer-file-select.open",
    )) {
      if (otherSelect !== select) {
        otherSelect.close?.();
      }
    }

    search.value = "";
    activeIndex = -1;
    renderOptions();
    setExpanded(true);

    requestAnimationFrame(() => {
      search.focus();
    });
  }

  function close() {
    search.value = "";
    activeIndex = -1;
    setExpanded(false);
  }

  select.close = close;

  select.setFiles = (
    newFiles,
    newPlaceholder,
  ) => {
    files = newFiles;
    selectedPath = "";
    currentPlaceholder = newPlaceholder;

    updateLabel();
    close();

    select.disabled = files.length === 0;
  };

  Object.defineProperty(select, "value", {
    get() {
      return selectedPath;
    },
    set(filePath) {
      selectedPath = filePath;
      updateLabel();
      close();
    },
  });

  Object.defineProperty(select, "disabled", {
    get() {
      return button.disabled;
    },
    set(disabled) {
      button.disabled = disabled;

      select.classList.toggle(
        "disabled",
        disabled,
      );

      if (disabled) {
        close();
      }
    },
  });

  button.addEventListener("click", () => {
    if (select.classList.contains("open")) {
      close();
      return;
    }

    open();
  });

  button.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown") {
      event.preventDefault();
      open();
    }
  });

  search.addEventListener("input", () => {
    activeIndex = -1;
    renderOptions();
  });

  search.addEventListener("keydown", (event) => {
    if (event.key === "Escape") {
      event.preventDefault();
      close();
      button.focus();
      return;
    }

    if (filteredFiles.length === 0) {
      return;
    }

    if (event.key === "ArrowDown") {
      event.preventDefault();

      activeIndex = Math.min(
        activeIndex + 1,
        filteredFiles.length - 1,
      );

      renderOptions();
      return;
    }

    if (event.key === "ArrowUp") {
      event.preventDefault();

      activeIndex =
        activeIndex <= 0
          ? filteredFiles.length - 1
          : activeIndex - 1;

      renderOptions();
      return;
    }

    if (event.key === "Enter") {
      event.preventDefault();

      const index =
        activeIndex >= 0 ? activeIndex : 0;

      const file = filteredFiles[index];

      if (file !== undefined) {
        selectFile(file);
      }
    }
  });

  document.addEventListener("click", (event) => {
    if (!select.contains(event.target)) {
      close();
    }
  });

  select.disabled = true;

  return select;
}

export { createFileSelect };
