import {
    describe,
    expect,
    it,
} from "vitest";

import {
    getFileStem,
} from "../../../tiatoolbox/visualization/openlayers/src/utils/paths.js";

describe("getFileStem", () => {
    it.each([
        [
            "/slides/CMU-1.svs",
            "CMU-1",
        ],
        [
            "C:\\slides\\CMU-1.svs",
            "CMU-1",
        ],
        [
            "/overlays/tissue_mask.png",
            "tissue_mask",
        ],
        [
            "/overlays/semantic_segmentation.db",
            "semantic_segmentation",
        ],
        [
            "/slides/no-extension",
            "no-extension",
        ],
        [
            "/slides/.hidden",
            ".hidden",
        ],
    ])(
        "returns the filename stem for %s",
        (filePath, expected) => {
            // Test getting a filename stem from supported path formats.
            expect(
                getFileStem(filePath),
            ).toBe(expected);
        },
    );
});
