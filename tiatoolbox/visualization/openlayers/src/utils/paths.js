// Return a filename without its final extension for POSIX or Windows paths.
function getFileStem(filePath) {
    const fileName = filePath.split(/[\\/]/).pop() ?? filePath;
    const extensionIndex = fileName.lastIndexOf(".");

    if (extensionIndex <= 0) {
        return fileName;
    }

    return fileName.slice(0, extensionIndex);
}

export { getFileStem };
