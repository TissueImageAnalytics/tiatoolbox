import { resolve } from "node:path";
import { fileURLToPath } from "node:url";

import { defineConfig } from "vite";

const currentDirectory = fileURLToPath(new URL(".", import.meta.url));

export default defineConfig({
  build: {
    emptyOutDir: false,
    lib: {
      entry: resolve(currentDirectory, "src/visualize_beta_viewer.js"),
      formats: ["es"],
      fileName: "visualize_beta_viewer",
      cssFileName: "visualize_beta_viewer",
    },
    outDir: resolve(
      currentDirectory,
      "../../data/visualization/static/openlayers",
    ),
  },
});
