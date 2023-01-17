// handle redirects
(() => {
    let anchorMap = {
        "installation": "installation.html",
        "getting-started": "getting_started.html#getting-started",
        "highlights": "getting_started.html#highlights",
        "contributing": "contributing.html",
        "projects-using-zarr": "getting_started.html#projects-using-zarr",
        "acknowledgments": "acknowledgments.html"
    }

    let hash = window.location.hash.substring(1);
    if (hash) {
        window.location.replace(anchorMap[hash]);
    }
})();
