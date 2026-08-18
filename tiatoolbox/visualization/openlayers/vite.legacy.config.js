import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";

const currentDirectory = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  build: {
    emptyOutDir: false,
    lib: {
      entry: resolve(currentDirectory, "src/main_legacy.js"),
      formats: ["es"],
      fileName: "viewer_legacy",
      cssFileName: "viewer_legacy",
    },
    outDir: resolve(
      currentDirectory,
      "../../data/visualization/static/openlayers",
    ),
  },
});
