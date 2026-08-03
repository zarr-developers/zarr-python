/* Material's header source widget shows the monorepo's latest release
   (e.g. v3.3.0). Replace it with the latest tag for this package, and
   patch the theme's sessionStorage cache so instant navigation keeps it.

   The GitHub repo and the package's tag prefix (e.g. zarr_metadata-) are
   derived from repo_url, whose final path segment is the package directory,
   so this file is identical across all sub-package docs sites. */
(() => {
  const source = document.querySelector('[data-md-component="source"]')
  if (!source || !source.href)
    return
  const segments = new URL(source.href).pathname.split("/").filter(Boolean)
  const REPO = segments.slice(0, 2).join("/")
  const PREFIX = segments[segments.length - 1].replace(/-/g, "_") + "-"

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

  /* Cache the tag for the tab's lifetime (like the theme's own __source
     cache) so navigation doesn't burn GitHub's unauthenticated API rate
     limit. Keyed by prefix in case sub-package sites share an origin. */
  const CACHE_KEY = `__package_tag/${PREFIX}`

  async function latestPackageTag() {
    const cached = sessionStorage.getItem(CACHE_KEY)
    if (cached)
      return cached
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
    if (versions[0])
      sessionStorage.setItem(CACHE_KEY, versions[0])
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
    new MutationObserver((_, observer) => {
      if (patchCache(version)) {
        patchDom(version)
        observer.disconnect()
      }
    }).observe(source, { childList: true, subtree: true })
  })
})()
