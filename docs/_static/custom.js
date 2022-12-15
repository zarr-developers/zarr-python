// handle redirects
(() => {
    let anchorMap = {
        "installation": "installation.html"
    }

    let hash = window.location.hash.substring(1);
    if (hash) {
        window.location.replace(anchorMap[hash]);
    }
})();