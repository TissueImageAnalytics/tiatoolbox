import {
    afterEach,
    beforeEach,
    describe,
    expect,
    it,
    vi,
} from "vitest";

import {
    createFileSelect,
} from "../../../tiatoolbox/visualization/openlayers/src/components/file-select.js";

const files = [
    {
        name: "CMU-1.svs",
        path: "/slides/CMU-1.svs",
    },
    {
        name: "CMU-2.svs",
        path: "/slides/CMU-2.svs",
    },
    {
        name: "Tumour Sample.svs",
        path: "/slides/Tumour Sample.svs",
    },
];

const originalScrollIntoView =
    HTMLElement.prototype.scrollIntoView;

function createPopulatedSelect(
    placeholder = "Select slide",
) {
    const select =
        createFileSelect(placeholder);

    document.body.append(select);

    select.setFiles(
        files,
        placeholder,
    );

    return select;
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

function getMenu(select) {
    return select.querySelector(
        ".viewer-file-select-menu",
    );
}

function getSearch(select) {
    return select.querySelector(
        ".viewer-file-select-search",
    );
}

function getOptions(select) {
    return [
        ...select.querySelectorAll(
            ".viewer-file-select-option",
        ),
    ];
}

function dispatchKey(element, key) {
    const event = new KeyboardEvent(
        "keydown",
        {
            key,
            bubbles: true,
            cancelable: true,
        },
    );

    element.dispatchEvent(event);

    return event;
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

    Object.defineProperty(
        HTMLElement.prototype,
        "scrollIntoView",
        {
            configurable: true,
            writable: true,
            value: vi.fn(),
        },
    );
});

afterEach(() => {
    document.body.replaceChildren();

    vi.unstubAllGlobals();

    if (
        originalScrollIntoView === undefined
    ) {
        delete HTMLElement.prototype
            .scrollIntoView;
    } else {
        Object.defineProperty(
            HTMLElement.prototype,
            "scrollIntoView",
            {
                configurable: true,
                writable: true,
                value: originalScrollIntoView,
            },
        );
    }
});

describe("createFileSelect", () => {
    it("creates an initially disabled selector", () => {
        const select =
            createFileSelect("Select slide");

        document.body.append(select);

        const button =
            getButton(select);
        const label =
            getLabel(select);
        const menu =
            getMenu(select);
        const search =
            getSearch(select);
        const options =
            select.querySelector(
                ".viewer-file-select-options",
            );

        expect(select.disabled).toBe(true);
        expect(
            select.classList.contains(
                "disabled",
            ),
        ).toBe(true);
        expect(button.disabled).toBe(true);
        expect(label.textContent).toBe(
            "Select slide",
        );
        expect(menu.hidden).toBe(true);
        expect(
            button.getAttribute(
                "aria-haspopup",
            ),
        ).toBe("listbox");
        expect(
            button.getAttribute(
                "aria-expanded",
            ),
        ).toBe("false");
        expect(
            button.getAttribute(
                "aria-controls",
            ),
        ).toBe(options.id);
        expect(
            search.getAttribute(
                "aria-controls",
            ),
        ).toBe(options.id);
        expect(
            search.getAttribute(
                "aria-label",
            ),
        ).toBe("Search select slide");
    });

    it("gives different selectors unique listbox IDs", () => {
        const first =
            createFileSelect("First");
        const second =
            createFileSelect("Second");

        const firstOptions =
            first.querySelector(
                ".viewer-file-select-options",
            );
        const secondOptions =
            second.querySelector(
                ".viewer-file-select-options",
            );

        expect(firstOptions.id).not.toBe(
            secondOptions.id,
        );
    });
});

describe("setFiles and value", () => {
    it("enables the selector and updates its placeholder", () => {
        const select =
            createFileSelect("Loading");

        document.body.append(select);

        select.setFiles(
            files,
            "Choose slide",
        );

        expect(select.disabled).toBe(false);
        expect(
            getButton(select).disabled,
        ).toBe(false);
        expect(
            getLabel(select).textContent,
        ).toBe("Choose slide");
        expect(select.value).toBe("");
    });

    it("resets and disables the selector when given no files", () => {
        const select =
            createPopulatedSelect();

        select.value =
            "/slides/CMU-1.svs";

        getButton(select).click();

        expect(
            select.classList.contains("open"),
        ).toBe(true);

        select.setFiles(
            [],
            "No slides available",
        );

        expect(select.value).toBe("");
        expect(select.disabled).toBe(true);
        expect(
            getLabel(select).textContent,
        ).toBe("No slides available");
        expect(
            select.classList.contains("open"),
        ).toBe(false);
        expect(
            getMenu(select).hidden,
        ).toBe(true);
    });

    it("sets a known value without dispatching a change event", () => {
        const select =
            createPopulatedSelect();
        const changeHandler = vi.fn();

        select.addEventListener(
            "change",
            changeHandler,
        );

        select.value =
            "/slides/CMU-2.svs";

        expect(select.value).toBe(
            "/slides/CMU-2.svs",
        );
        expect(
            getLabel(select).textContent,
        ).toBe("CMU-2.svs");
        expect(
            getLabel(select).title,
        ).toBe("/slides/CMU-2.svs");
        expect(
            changeHandler,
        ).not.toHaveBeenCalled();
    });

    it("uses the filename for an unknown selected path", () => {
        const select =
            createPopulatedSelect();

        select.value =
            "C:\\external\\other-slide.svs";

        expect(
            getLabel(select).textContent,
        ).toBe("other-slide.svs");
        expect(
            getLabel(select).title,
        ).toBe(
            "C:\\external\\other-slide.svs",
        );
    });
});

describe("opening and closing", () => {
    it("opens and closes when the button is clicked", () => {
        const select =
            createPopulatedSelect();
        const button =
            getButton(select);
        const menu =
            getMenu(select);

        button.click();

        expect(
            select.classList.contains("open"),
        ).toBe(true);
        expect(menu.hidden).toBe(false);
        expect(
            button.getAttribute(
                "aria-expanded",
            ),
        ).toBe("true");
        expect(document.activeElement).toBe(
            getSearch(select),
        );

        button.click();

        expect(
            select.classList.contains("open"),
        ).toBe(false);
        expect(menu.hidden).toBe(true);
        expect(
            button.getAttribute(
                "aria-expanded",
            ),
        ).toBe("false");
    });

    it("does not open while disabled", () => {
        const select =
            createPopulatedSelect();

        select.disabled = true;

        getButton(select).click();

        expect(
            select.classList.contains("open"),
        ).toBe(false);
        expect(
            getMenu(select).hidden,
        ).toBe(true);
    });

    it("does not open from the keyboard while disabled", () => {
        const select =
            createPopulatedSelect();

        select.disabled = true;

        const event =
            dispatchKey(
                getButton(select),
                "ArrowDown",
            );

        expect(
            event.defaultPrevented,
        ).toBe(true);

        expect(
            select.classList.contains("open"),
        ).toBe(false);

        expect(
            getMenu(select).hidden,
        ).toBe(true);
    });

    it("closes when disabled while open", () => {
        const select =
            createPopulatedSelect();

        getButton(select).click();

        expect(
            select.classList.contains("open"),
        ).toBe(true);

        select.disabled = true;

        expect(
            select.classList.contains("open"),
        ).toBe(false);
        expect(
            getMenu(select).hidden,
        ).toBe(true);
    });

    it("opens with ArrowDown from the button", () => {
        const select =
            createPopulatedSelect();
        const button =
            getButton(select);

        const event =
            dispatchKey(
                button,
                "ArrowDown",
            );

        expect(event.defaultPrevented).toBe(
            true,
        );
        expect(
            select.classList.contains("open"),
        ).toBe(true);
        expect(document.activeElement).toBe(
            getSearch(select),
        );
    });

    it("closes another selector when opened", () => {
        const first =
            createPopulatedSelect("First");
        const second =
            createPopulatedSelect("Second");

        getButton(first).click();

        expect(
            first.classList.contains("open"),
        ).toBe(true);

        getButton(second).click();

        expect(
            first.classList.contains("open"),
        ).toBe(false);
        expect(
            second.classList.contains("open"),
        ).toBe(true);
    });

    it("closes when clicking outside the selector", () => {
        const select =
            createPopulatedSelect();
        const outside =
            document.createElement("button");

        document.body.append(outside);

        getButton(select).click();

        expect(
            select.classList.contains("open"),
        ).toBe(true);

        outside.click();

        expect(
            select.classList.contains("open"),
        ).toBe(false);
    });

    it("can be closed through its close method", () => {
        const select =
            createPopulatedSelect();

        getButton(select).click();

        select.close();

        expect(
            select.classList.contains("open"),
        ).toBe(false);
        expect(
            getMenu(select).hidden,
        ).toBe(true);
    });

    it("prevents mousedown from moving focus away from an option", () => {
        const select =
            createPopulatedSelect();

        getButton(select).click();

        const option =
            select.querySelector(
                ".viewer-file-select-option",
            );

        expect(option).not.toBeNull();

        const event =
            new MouseEvent(
                "mousedown",
                {
                    bubbles: true,
                    cancelable: true,
                },
            );

        option.dispatchEvent(event);

        expect(
            event.defaultPrevented,
        ).toBe(true);
    });
});

describe("filtering", () => {
    it("filters files case-insensitively and ignores surrounding whitespace", () => {
        const select =
            createPopulatedSelect();

        getButton(select).click();

        const search =
            getSearch(select);

        search.value = "  tumour  ";
        search.dispatchEvent(
            new Event(
                "input",
                {
                    bubbles: true,
                },
            ),
        );

        expect(
            getOptions(select).map(
                (option) =>
                    option.textContent,
            ),
        ).toEqual([
            "Tumour Sample.svs",
        ]);
    });

    it("shows an empty state when no files match", () => {
        const select =
            createPopulatedSelect();

        getButton(select).click();

        const search =
            getSearch(select);

        search.value = "missing";
        search.dispatchEvent(
            new Event(
                "input",
                {
                    bubbles: true,
                },
            ),
        );

        expect(
            getOptions(select),
        ).toHaveLength(0);
        expect(
            select.querySelector(
                ".viewer-file-select-empty",
            ).textContent,
        ).toBe("No matches");

        dispatchKey(
            search,
            "Enter",
        );

        expect(select.value).toBe("");
    });
});

describe("selection", () => {
    it("selects a file by clicking an option", () => {
        const select =
            createPopulatedSelect();
        const changeHandler = vi.fn();

        select.addEventListener(
            "change",
            changeHandler,
        );

        getButton(select).click();

        getOptions(select)[1].click();

        expect(select.value).toBe(
            "/slides/CMU-2.svs",
        );
        expect(
            getLabel(select).textContent,
        ).toBe("CMU-2.svs");
        expect(
            getLabel(select).title,
        ).toBe("/slides/CMU-2.svs");
        expect(
            select.classList.contains("open"),
        ).toBe(false);
        expect(
            changeHandler,
        ).toHaveBeenCalledOnce();
        expect(
            changeHandler.mock.calls[0][0]
                .detail,
        ).toBe("/slides/CMU-2.svs");

        getButton(select).click();

        const selectedOption =
            getOptions(select)[1];

        expect(
            selectedOption.classList.contains(
                "selected",
            ),
        ).toBe(true);
        expect(
            selectedOption.getAttribute(
                "aria-selected",
            ),
        ).toBe("true");
    });

    it("selects the first filtered file with Enter when none is active", () => {
        const select =
            createPopulatedSelect();

        getButton(select).click();

        dispatchKey(
            getSearch(select),
            "Enter",
        );

        expect(select.value).toBe(
            "/slides/CMU-1.svs",
        );
    });

    it("moves down through options and selects the active file", () => {
        const select =
            createPopulatedSelect();

        getButton(select).click();

        const search =
            getSearch(select);

        dispatchKey(
            search,
            "ArrowDown",
        );
        dispatchKey(
            search,
            "ArrowDown",
        );

        expect(
            select.querySelector(
                ".viewer-file-select-option.active",
            ).textContent,
        ).toBe("CMU-2.svs");

        dispatchKey(
            search,
            "Enter",
        );

        expect(select.value).toBe(
            "/slides/CMU-2.svs",
        );
    });

    it("does not move past the final option with ArrowDown", () => {
        const select =
            createPopulatedSelect();

        getButton(select).click();

        const search =
            getSearch(select);

        dispatchKey(
            search,
            "ArrowDown",
        );
        dispatchKey(
            search,
            "ArrowDown",
        );
        dispatchKey(
            search,
            "ArrowDown",
        );
        dispatchKey(
            search,
            "ArrowDown",
        );

        expect(
            select.querySelector(
                ".viewer-file-select-option.active",
            ).textContent,
        ).toBe("Tumour Sample.svs");
    });

    it("wraps to the final option with ArrowUp", () => {
        const select =
            createPopulatedSelect();

        getButton(select).click();

        const search =
            getSearch(select);

        dispatchKey(
            search,
            "ArrowUp",
        );

        expect(
            select.querySelector(
                ".viewer-file-select-option.active",
            ).textContent,
        ).toBe("Tumour Sample.svs");

        dispatchKey(
            search,
            "Enter",
        );

        expect(select.value).toBe(
            "/slides/Tumour Sample.svs",
        );
    });

    it("closes with Escape and returns focus to the button", () => {
        const select =
            createPopulatedSelect();
        const button =
            getButton(select);

        button.click();

        const search =
            getSearch(select);

        const event =
            dispatchKey(
                search,
                "Escape",
            );

        expect(event.defaultPrevented).toBe(
            true,
        );
        expect(
            select.classList.contains("open"),
        ).toBe(false);
        expect(document.activeElement).toBe(
            button,
        );
    });
});
