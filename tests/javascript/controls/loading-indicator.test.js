import {
    describe,
    expect,
    it,
    vi,
} from "vitest";

import {
    createLoadingIndicatorController,
} from "../../../tiatoolbox/visualization/openlayers/src/controls/loading-indicator.js";

function createHarness() {
    const handlers = new Map();

    const map = {
        on: vi.fn((event, handler) => {
            handlers.set(event, handler);
        }),

        render: vi.fn(),
    };

    const element =
        document.createElement("div");

    const onChange = vi.fn();

    const controller =
        createLoadingIndicatorController({
            map,
            element,
            onChange,
        });

    function emit(event) {
        handlers.get(event)?.();
    }

    return {
        controller,
        element,
        map,
        onChange,
        emit,
    };
}

describe("loading indicator", () => {
    it("starts in the loading state", () => {
        const {
            controller,
            element,
            map,
            onChange,
        } = createHarness();

        expect(
            controller.isLoading(),
        ).toBe(true);

        expect(element.hidden).toBe(false);

        expect(
            onChange,
        ).toHaveBeenLastCalledWith(true);

        expect(map.render).toHaveBeenCalledOnce();
    });

    it("tracks map loading and rendering", () => {
        const {
            controller,
            element,
            onChange,
            emit,
        } = createHarness();

        emit("rendercomplete");

        expect(
            controller.isLoading(),
        ).toBe(false);

        expect(element.hidden).toBe(true);

        expect(
            onChange,
        ).toHaveBeenLastCalledWith(false);

        emit("loadstart");

        expect(
            controller.isLoading(),
        ).toBe(true);

        expect(element.hidden).toBe(false);

        expect(
            onChange,
        ).toHaveBeenLastCalledWith(true);
    });

    it("tracks pending viewer operations", () => {
        const {
            controller,
            map,
            emit,
        } = createHarness();

        emit("rendercomplete");

        controller.begin();

        expect(
            controller.isLoading(),
        ).toBe(true);

        emit("rendercomplete");

        expect(
            controller.isLoading(),
        ).toBe(true);

        controller.end();

        expect(
            controller.isLoading(),
        ).toBe(true);

        expect(map.render).toHaveBeenCalledTimes(2);

        emit("rendercomplete");

        expect(
            controller.isLoading(),
        ).toBe(false);
    });

    it("tracks multiple pending operations", () => {
        const {
            controller,
            map,
            emit,
        } = createHarness();

        emit("rendercomplete");

        controller.begin();
        controller.begin();

        controller.end();

        expect(
            controller.isLoading(),
        ).toBe(true);

        expect(map.render).toHaveBeenCalledOnce();

        controller.end();

        expect(map.render).toHaveBeenCalledTimes(2);
    });

    it("does not allow pending operations to become negative", () => {
        const {
            controller,
            map,
        } = createHarness();

        controller.end();

        expect(
            controller.isLoading(),
        ).toBe(true);

        expect(map.render).toHaveBeenCalledTimes(2);
    });

    it("works without a loading callback", () => {
        const handlers = new Map();

        const map = {
            on(event, handler) {
                handlers.set(event, handler);
            },

            render: vi.fn(),
        };

        const element =
            document.createElement("div");

        expect(() => {
            createLoadingIndicatorController({
                map,
                element,
            });
        }).not.toThrow();
    });
});
