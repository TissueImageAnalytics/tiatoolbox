import {
    afterEach,
    beforeEach,
    describe,
    expect,
    it,
    vi,
} from "vitest";

import {
    createOverviewMapController,
} from "../../../tiatoolbox/visualization/openlayers/src/controls/overview-map.js";

class ResizeObserverMock {
    constructor(callback) {
        this.callback = callback;
    }

    observe() {}

    unobserve() {}

    disconnect() {}
}

function createCheckbox(
    checked = true,
) {
    const input =
        document.createElement(
            "input",
        );

    input.type = "checkbox";
    input.checked = checked;

    return input;
}

function createSelect(
    values,
    value,
) {
    const select =
        document.createElement(
            "select",
        );

    for (const optionValue of values) {
        const option =
            document.createElement(
                "option",
            );

        option.value = optionValue;
        option.textContent =
            optionValue;

        select.append(option);
    }

    select.value = value;

    return select;
}

function createProjection() {
    return "EPSG:3857";
}

function createSource({
    extent = [
        0,
        0,
        600,
        300,
    ],
} = {}) {
    const source =
        new EventTarget();

    source.getState =
        vi.fn(() => "ready");

    source.getTileGrid =
        vi.fn(() => ({
            getExtent: vi.fn(
                () => extent,
            ),
        }));

    return source;
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

function createHarness({
    source = null,
    extent = [
        0,
        0,
        600,
        300,
    ],
    size = "default",
    visible = true,
    slideLoaded = true,
} = {}) {
    const projection =
        createProjection();

    const state = {
        slideLoaded,
    };

    const mapView = {
        getProjection: vi.fn(
            () => projection,
        ),
    };

    const map = {
        addControl: vi.fn(
            (control) => {
                const overviewMap =
                    control.getOverviewMap();

                vi.spyOn(
                    overviewMap,
                    "updateSize",
                ).mockImplementation(
                    () => {},
                );

                vi.spyOn(
                    overviewMap,
                    "renderSync",
                ).mockImplementation(
                    () => {},
                );
            },
        ),

        getView: vi.fn(
            () => mapView,
        ),
    };

    const sizeSelect =
        createSelect(
            [
                "small",
                "default",
                "large",
            ],
            size,
        );

    const visibleInput =
        createCheckbox(visible);

    document.body.append(
        sizeSelect,
        visibleInput,
    );

    const controller =
        createOverviewMapController({
            map,
            source,
            projection,
            extent,
            sizeSelect,
            visibleInput,
            hasSlide: () =>
                state.slideLoaded,
        });

    const control =
        controller.control;

    const overviewMap =
        control.getOverviewMap();

    overviewMap.updateSize.mockClear();
    overviewMap.renderSync.mockClear();

    return {
        state,
        map,
        projection,
        extent,
        sizeSelect,
        visibleInput,
        controller,
        control,
        overviewMap,
    };
}

function getOverviewLayer(
    control,
) {
    return control
        .getOverviewMap()
        .getLayers()
        .item(0);
}

beforeEach(() => {
    document.body.replaceChildren();

    vi.stubGlobal(
        "ResizeObserver",
        ResizeObserverMock,
    );

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
    it("creates and adds the overview map control", () => {
        const {
            map,
            control,
        } = createHarness();

        expect(control).toBeDefined();

        expect(
            map.addControl,
        ).toHaveBeenCalledOnce();

        expect(
            map.addControl,
        ).toHaveBeenCalledWith(
            control,
        );

        expect(
            control.getCollapsed(),
        ).toBe(false);

        expect(
            control.getCollapsible(),
        ).toBe(true);

        expect(
            control.element.classList
                .contains(
                    "ol-custom-overviewmap",
                ),
        ).toBe(true);
    });

    it("creates an overview layer without a source when none is supplied", () => {
        const {
            control,
        } = createHarness();

        expect(
            getOverviewLayer(
                control,
            ).getSource(),
        ).toBeNull();
    });

    it("uses a supplied source for the overview layer", () => {
        const source =
            createSource();

        const {
            control,
        } = createHarness({
            source,
        });

        expect(
            getOverviewLayer(
                control,
            ).getSource(),
        ).toBe(source);
    });

    it.each([
        [
            "small",
            220,
            180,
            600 / 220,
        ],
        [
            "default",
            300,
            250,
            2,
        ],
        [
            "large",
            380,
            320,
            600 / 380,
        ],
    ])(
        "uses the %s overview map size",
        (
            size,
            expectedWidth,
            expectedHeight,
            expectedResolution,
        ) => {
            const {
                control,
                overviewMap,
            } = createHarness({
                size,
            });

            expect(
                control.element.style
                    .getPropertyValue(
                        "--overview-map-width",
                    ),
            ).toBe(
                `${expectedWidth}px`,
            );

            expect(
                control.element.style
                    .getPropertyValue(
                        "--overview-map-height",
                    ),
            ).toBe(
                `${expectedHeight}px`,
            );

            expect(
                overviewMap
                    .getView()
                    .getCenter(),
            ).toEqual([
                300,
                150,
            ]);

            expect(
                overviewMap
                    .getView()
                    .getResolution(),
            ).toBeCloseTo(
                expectedResolution,
            );
        },
    );

    it("falls back to the default size for an unknown value", () => {
        const {
            control,
            overviewMap,
        } = createHarness({
            size: "",
        });

        expect(
            control.element.style
                .getPropertyValue(
                    "--overview-map-width",
                ),
        ).toBe("300px");

        expect(
            control.element.style
                .getPropertyValue(
                    "--overview-map-height",
                ),
        ).toBe("250px");

        expect(
            overviewMap
                .getView()
                .getResolution(),
        ).toBeCloseTo(2);
    });

    it("creates the overview toggle labels", () => {
        const {
            control,
        } = createHarness();

        expect(
            control.element.querySelector(
                ".overview-toggle-icon",
            ),
        ).not.toBeNull();

        expect(
            control.element.innerHTML,
        ).toContain(
            "fa-chevron-up",
        );
    });
});

describe("source and view", () => {
    it("updates the overview layer source", () => {
        const {
            controller,
            control,
        } = createHarness();

        const source =
            createSource();

        controller.setSource(
            source,
        );

        expect(
            getOverviewLayer(
                control,
            ).getSource(),
        ).toBe(source);

        controller.setSource(
            null,
        );

        expect(
            getOverviewLayer(
                control,
            ).getSource(),
        ).toBeNull();
    });

    it("sets a new overview view", () => {
        const {
            controller,
            overviewMap,
            projection,
        } = createHarness();

        controller.setView(
            projection,
            [
                0,
                0,
                100,
                200,
            ],
        );

        const view =
            overviewMap.getView();

        expect(
            view.getCenter(),
        ).toEqual([
            50,
            100,
        ]);

        expect(
            view.getResolution(),
        ).toBeCloseTo(0.8);

        expect(
            view.getResolutions(),
        ).toEqual([
            0.8,
        ]);
    });

    it("keeps the overview view centred on its extent", () => {
        const {
            controller,
            overviewMap,
            projection,
        } = createHarness();

        controller.setView(
            projection,
            [
                0,
                0,
                100,
                200,
            ],
        );

        const view =
            overviewMap.getView();

        view.setCenter([
            10,
            20,
        ]);

        expect(
            view.getCenter(),
        ).toEqual([
            50,
            100,
        ]);
    });

    it("rebuilds the overview view when the size changes and a source exists", () => {
        const source =
            createSource({
                extent: [
                    0,
                    0,
                    600,
                    300,
                ],
            });

        const {
            sizeSelect,
            control,
            overviewMap,
        } = createHarness({
            source,
        });

        const updateSize =
            overviewMap.updateSize;

        const renderSync =
            overviewMap.renderSync;

        const setView =
            vi.spyOn(
                overviewMap,
                "setView",
            );

        sizeSelect.value =
            "small";

        dispatchChange(
            sizeSelect,
        );

        expect(
            control.element.style
                .getPropertyValue(
                    "--overview-map-width",
                ),
        ).toBe("220px");

        expect(
            control.element.style
                .getPropertyValue(
                    "--overview-map-height",
                ),
        ).toBe("180px");

        expect(
            updateSize,
        ).toHaveBeenCalledOnce();

        expect(
            setView,
        ).toHaveBeenCalledOnce();

        expect(
            overviewMap
                .getView()
                .getResolution(),
        ).toBeCloseTo(
            600 / 220,
        );

        expect(
            renderSync,
        ).toHaveBeenCalledOnce();
    });

    it("does not replace the view during size updates without a source", () => {
        const {
            controller,
            overviewMap,
        } = createHarness();

        const setView =
            vi.spyOn(
                overviewMap,
                "setView",
            );

        const updateSize =
            overviewMap.updateSize;

        const renderSync =
            overviewMap.renderSync;

        controller.updateSize();

        expect(
            updateSize,
        ).toHaveBeenCalledOnce();

        expect(
            setView,
        ).not.toHaveBeenCalled();

        expect(
            renderSync,
        ).toHaveBeenCalledOnce();
    });
});

describe("visibility", () => {
    it("shows and refreshes when a slide is available and the control is enabled", () => {
        const {
            controller,
            control,
            overviewMap,
        } = createHarness();

        const updateSize =
            overviewMap.updateSize;

        const renderSync =
            overviewMap.renderSync;

        controller.updateVisibility();

        expect(
            control.element.classList
                .contains(
                    "viewer-control-hidden",
                ),
        ).toBe(false);

        expect(
            updateSize,
        ).toHaveBeenCalledOnce();

        expect(
            renderSync,
        ).toHaveBeenCalledOnce();
    });

    it("hides when no slide is available", () => {
        const {
            controller,
            control,
            overviewMap,
        } = createHarness({
            slideLoaded: false,
        });

        const updateSize =
            overviewMap.updateSize;

        controller.updateVisibility();

        expect(
            control.element.classList
                .contains(
                    "viewer-control-hidden",
                ),
        ).toBe(true);

        expect(
            updateSize,
        ).not.toHaveBeenCalled();
    });

    it("hides when overview map visibility is disabled", () => {
        const {
            controller,
            control,
        } = createHarness({
            visible: false,
        });

        controller.updateVisibility();

        expect(
            control.element.classList
                .contains(
                    "viewer-control-hidden",
                ),
        ).toBe(true);
    });

    it("respects viewer enabled state", () => {
        const {
            controller,
            control,
            overviewMap,
        } = createHarness();

        const updateSize =
            overviewMap.updateSize;

        const renderSync =
            overviewMap.renderSync;

        controller.setViewerEnabled(
            false,
        );

        expect(
            control.element.classList
                .contains(
                    "viewer-control-hidden",
                ),
        ).toBe(true);

        expect(
            updateSize,
        ).not.toHaveBeenCalled();

        controller.setViewerEnabled(
            true,
        );

        expect(
            control.element.classList
                .contains(
                    "viewer-control-hidden",
                ),
        ).toBe(false);

        expect(
            updateSize,
        ).toHaveBeenCalledOnce();

        expect(
            renderSync,
        ).toHaveBeenCalledOnce();
    });

    it("remains hidden when viewer is enabled but the setting is disabled", () => {
        const {
            controller,
            control,
            overviewMap,
        } = createHarness({
            visible: false,
        });

        const updateSize =
            overviewMap.updateSize;

        const renderSync =
            overviewMap.renderSync;

        controller.setViewerEnabled(
            true,
        );

        expect(
            control.element.classList
                .contains(
                    "viewer-control-hidden",
                ),
        ).toBe(true);

        expect(
            updateSize,
        ).toHaveBeenCalledOnce();

        expect(
            renderSync,
        ).toHaveBeenCalledOnce();
    });
});

describe("refresh", () => {
    it("updates and renders the internal overview map", () => {
        const {
            controller,
            overviewMap,
        } = createHarness();

        const updateSize =
            overviewMap.updateSize;

        const renderSync =
            overviewMap.renderSync;

        controller.refresh();

        expect(
            updateSize,
        ).toHaveBeenCalledOnce();

        expect(
            renderSync,
        ).toHaveBeenCalledOnce();
    });
});
