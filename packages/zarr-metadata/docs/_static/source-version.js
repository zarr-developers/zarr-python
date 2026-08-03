/* Material's header source widget shows the monorepo's latest release
   (e.g. v3.3.0). Replace it with the latest tag for this package, and
   patch the theme's sessionStorage cache so instant navigation keeps it. */
(() => {
  const REPO = "zarr-developers/zarr-python"
  const PREFIX = "zarr_metadata-"

  const parse = version => version
    .replace(/^v/, "")
    .split(/[.+-]/)
    .map(part => parseInt(part, 10) || 0)

  const newestFirst = (a, b) => {
    const va = parse(a), vb = parse(b)
    for (let i = 0; i < Math.max(va.length, vb.length); i++)
      if ((vb[i] || 0) !== (va[i] || 0))
        return (vb[i] || 0) - (va[i] || 0)
    return 0
  }

  async function latestPackageTag() {
    const response = await fetch(
      `https://api.github.com/repos/${REPO}/tags?per_page=100`
    )
    if (!response.ok)
      return undefined
    const tags = await response.json()
    const versions = tags
      .map(tag => tag.name)
      .filter(name => name.startsWith(PREFIX))
      .map(name => name.slice(PREFIX.length))
      .sort(newestFirst)
    return versions[0]
  }

  function patchDom(version) {
    for (const el of document.querySelectorAll(".md-source__fact--version"))
      el.textContent = version
  }

  function patchCache(version) {
    const facts = __md_get("__source", sessionStorage)
    if (!facts)
      return false
    facts.version = version
    __md_set("__source", facts, sessionStorage)
    return true
  }

  latestPackageTag().then(version => {
    if (!version)
      return
    patchDom(version)
    if (patchCache(version))
      return
    /* The theme's own API request hasn't resolved yet — patch both the
       cache and the DOM once it renders the facts list. */
    const source = document.querySelector('[data-md-component="source"]')
    if (!source)
      return
    new MutationObserver((_, observer) => {
      if (patchCache(version)) {
        patchDom(version)
        observer.disconnect()
      }
    }).observe(source, { childList: true, subtree: true })
  })
})()
