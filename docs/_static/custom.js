// handle redirects
(() => {
    let anchorMap = {
        "installation": "installation.html",
        "getting-started": "getting_started.html#getting-started",
        "highlights": "getting_started.html#highlights",
        "contributing": "contributing.html",
        "projects-using-zarr": "getting_started.html#projects-using-zarr",
        "acknowledgments": "acknowledgments.html",
        "contents": "getting_started.html#contents",
        "indices-and-tables": "api.html#indices-and-tables"
    }

    let hash = window.location.hash.substring(1);
    if (hash && hash in anchorMap) {
            window.location.replace(anchorMap[hash]);
    }
})();
