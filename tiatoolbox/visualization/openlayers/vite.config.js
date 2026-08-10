import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";

const currentDirectory = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  build: {
    emptyOutDir: true,
    lib: {
      entry: resolve(currentDirectory, "src/main.js"),
      formats: ["es"],
      fileName: "viewer",
      cssFileName: "viewer",
    },
    outDir: resolve(
      currentDirectory,
      "../../data/visualization/static/openlayers",
    ),
  },
});
