import { defineConfig } from "vitest/config";

export default defineConfig({
    server: {
        fs: {
            allow: ["../../.."],
        },
    },

    test: {
        environment: "jsdom",
        dir: "../../../tests/openlayers",
        include: ["**/*.test.js"],
        clearMocks: true,
        restoreMocks: true,

        coverage: {
            provider: "v8",
            reporter: ["text", "lcov"],

            include: [
                "src/api/**/*.js",
                "src/components/**/*.js",
                "src/controls/**/*.js",
                "src/panels/**/*.js",
                "src/utils/**/*.js",
            ],
        },
    },
});
