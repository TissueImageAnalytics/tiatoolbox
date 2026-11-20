import js from "@eslint/js";
import vitest from "@vitest/eslint-plugin";
import { defineConfig } from "eslint/config";
import globals from "globals";

const browserFiles = [
    "tiatoolbox/visualization/openlayers/src/**/*.js",
];

const nodeFiles = [
    "tiatoolbox/visualization/openlayers/eslint.config.js",
    "tiatoolbox/visualization/openlayers/vite.config.js",
    "tiatoolbox/visualization/openlayers/vite.legacy.config.js",
    "tiatoolbox/visualization/openlayers/vitest.config.js",
];

const testFiles = [
    "tests/javascript/**/*.js",
];

export default defineConfig([
    {
        name: "openlayers/javascript",
        files: [
            ...browserFiles,
            ...nodeFiles,
            ...testFiles,
        ],

        rules: {
            ...js.configs.recommended.rules,
        },
    },

    {
        name: "openlayers/browser",
        files: browserFiles,

        languageOptions: {
            globals: {
                ...globals.browser,
            },
        },
    },

    {
        name: "openlayers/node",
        files: nodeFiles,

        languageOptions: {
            globals: {
                ...globals.nodeBuiltin,
            },
        },
    },

    {
        name: "openlayers/tests",
        files: testFiles,

        plugins: {
            vitest,
        },

        languageOptions: {
            globals: {
                ...globals.browser,
                ...globals.nodeBuiltin,
            },
        },

        rules: {
            ...vitest.configs.recommended.rules,
        },
    },

    {
        linterOptions: {
            reportUnusedDisableDirectives: "error",
        },
    },
]);
