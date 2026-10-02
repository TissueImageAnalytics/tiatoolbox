import {
    describe,
    expect,
    it,
} from "vitest";

import {
    getAnnotationFilter,
} from "../../../tiatoolbox/visualization/openlayers/src/utils/annotation-filters.js";

describe("getAnnotationFilter", () => {
    it("returns no filter when all types are visible", () => {
        expect(
            getAnnotationFilter(
                [
                    "Tumour",
                    "Stroma",
                ],
                new Map(),
            ),
        ).toBeNull();
    });

    it("filters to visible annotation types", () => {
        const visibility =
            new Map([
                [
                    0,
                    false,
                ],
                [
                    "Tumour",
                    true,
                ],
                [
                    2,
                    true,
                ],
            ]);

        expect(
            getAnnotationFilter(
                [
                    0,
                    "Tumour",
                    2,
                ],
                visibility,
            ),
        ).toBe(
            '(props["type"]=="Tumour") | (props["type"]==2)',
        );
    });

    it("returns an impossible filter when all types are hidden", () => {
        const visibility =
            new Map([
                [
                    "None",
                    false,
                ],
            ]);

        expect(
            getAnnotationFilter(
                [
                    "None",
                ],
                visibility,
            ),
        ).toBe(
            '(props["type"]=="None") & (props["type"]!="None")',
        );
    });
});
