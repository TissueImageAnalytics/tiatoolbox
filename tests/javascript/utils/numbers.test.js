import {
    describe,
    expect,
    it,
} from "vitest";

import {
    getFiniteNumberRange,
} from "../../../tiatoolbox/visualization/openlayers/src/utils/numbers.js";

describe("getFiniteNumberRange", () => {
    it("returns the range of finite numbers", () => {
        expect(
            getFiniteNumberRange([
                0.5,
                -2,
                4,
                1.5,
            ]),
        ).toEqual([
            -2,
            4,
        ]);
    });

    it("returns the same value for a constant range", () => {
        expect(
            getFiniteNumberRange([
                0.5,
                0.5,
                0.5,
            ]),
        ).toEqual([
            0.5,
            0.5,
        ]);
    });

    it("rejects empty or non-finite values", () => {
        expect(
            getFiniteNumberRange([]),
        ).toBeNull();

        expect(
            getFiniteNumberRange([
                0.5,
                Number.NaN,
            ]),
        ).toBeNull();

        expect(
            getFiniteNumberRange([
                0.5,
                Infinity,
            ]),
        ).toBeNull();

        expect(
            getFiniteNumberRange([
                0.5,
                "1",
            ]),
        ).toBeNull();
    });

    it("handles large property value arrays", () => {
        const values =
            Array.from(
                {
                    length: 200000,
                },
                (_, index) =>
                    index,
            );

        expect(
            getFiniteNumberRange(
                values,
            ),
        ).toEqual([
            0,
            199999,
        ]);
    });
});
