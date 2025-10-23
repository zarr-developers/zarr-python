# /// script
# requires-python = ">=3.11"
# dependencies = [
#     "requests",
#     "beautifulsoup4",
#     "urllib3",
# ]
# ///

"""
Validate that all subpages from zarr stable docs exist in latest docs.
This script crawls the stable documentation and checks if each page
has a corresponding valid page in the latest documentation.

Generated using Claude
"""

import time
from collections import deque
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

STABLE_BASE = "https://zarr.readthedocs.io/en/stable"
LATEST_BASE = "https://zarr.readthedocs.io/en/latest"

class DocumentationValidator:
    def __init__(self, stable_base: str, latest_base: str) -> None:
        self.stable_base = stable_base.rstrip('/')
        self.latest_base = latest_base.rstrip('/')
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Documentation Validator)'
        })

    def get_relative_path(self, url: str, base: str) -> str:
        """Extract the relative path from a full URL."""
        if url.startswith(base):
            path = url[len(base):]
            # Remove fragment identifiers
            if '#' in path:
                path = path.split('#')[0]
            return path
        return ""

    def is_valid_doc_url(self, url: str, base: str) -> bool:
        """Check if URL is part of the documentation."""
        if not url.startswith(('http://', 'https://')):
            return False
        parsed = urlparse(url)
        base_parsed = urlparse(base)
        # Must be same domain and start with base path
        return (parsed.netloc == base_parsed.netloc and
                url.startswith(base))

    def fetch_page(self, url: str) -> tuple[int, str]:
        """Fetch a page and return status code and content."""
        try:
            response = self.session.get(url, timeout=10, allow_redirects=True)
        except requests.RequestException as e:
            print(f"  ✗ Error fetching {url}: {e}")
            return 0, ""
        return response.status_code, response.text

    def extract_links(self, html: str, base_url: str) -> set[str]:
        """Extract all documentation links from HTML."""
        soup = BeautifulSoup(html, 'html.parser')
        links = set()

        for a_tag in soup.find_all('a', href=True):
            href = a_tag['href']
            full_url = urljoin(base_url, href)

            # Remove fragment identifiers for deduplication
            if '#' in full_url:
                full_url = full_url.split('#')[0]

            if self.is_valid_doc_url(full_url, self.stable_base):
                links.add(full_url)

        return links

    def crawl_stable_docs(self) -> set[str]:
        """Crawl all pages in stable documentation."""
        print(f"🔍 Crawling stable documentation: {self.stable_base}")
        visited = set()
        to_visit = deque([self.stable_base + "/"])

        while to_visit:
            url = to_visit.popleft()

            if url in visited:
                continue

            visited.add(url)
            print(f"  Crawling: {url}")

            status_code, html = self.fetch_page(url)

            if status_code != 200 or not html:
                continue

            # Extract and queue new links
            links = self.extract_links(html, url)
            for link in links:
                if link not in visited:
                    to_visit.append(link)

            # Be respectful with rate limiting
            time.sleep(0.1)

        print(f"✓ Found {len(visited)} pages in stable docs\n")
        return visited

    def validate_latest_docs(self, stable_urls: set[str]) -> dict[str, list[str]]:
        """Check if all stable URLs exist in latest docs."""
        print(f"🔍 Validating pages in latest documentation: {self.latest_base}")

        results = {
            'valid': [],
            'missing': [],
            'error': []
        }

        for stable_url in sorted(stable_urls):
            relative_path = self.get_relative_path(stable_url, self.stable_base)
            latest_url = self.latest_base + relative_path

            print(f"  Checking: {relative_path}")
            status_code, _ = self.fetch_page(latest_url)

            if status_code == 200:
                results['valid'].append(relative_path)
                print("    ✓ Valid (200)")
            elif status_code == 404:
                results['missing'].append(relative_path)
                print("    ✗ Missing (404)")
            else:
                results['error'].append(f"{relative_path} (status: {status_code})")
                print(f"    ⚠ Error (status: {status_code})")

            time.sleep(0.1)

        return results

    def print_summary(self, results: dict[str, list[str]]) -> None:
        """Print validation summary."""
        print("\n" + "="*70)
        print("VALIDATION SUMMARY")
        print("="*70)

        total = len(results['valid']) + len(results['missing']) + len(results['error'])

        print(f"\n✓ Valid pages: {len(results['valid'])}/{total}")
        print(f"✗ Missing pages: {len(results['missing'])}/{total}")
        print(f"⚠ Error pages: {len(results['error'])}/{total}")

        if results['missing']:
            print("\n" + "-"*70)
            print("MISSING PAGES:")
            print("-"*70)
            for path in results['missing']:
                print(f"  • {path}")

        if results['error']:
            print("\n" + "-"*70)
            print("ERROR PAGES:")
            print("-"*70)
            for info in results['error']:
                print(f"  • {info}")

        print("\n" + "="*70)

        if not results['missing'] and not results['error']:
            print("🎉 All pages validated successfully!")
        else:
            print(f"⚠️  {len(results['missing']) + len(results['error'])} issues found")
        print("="*70)

def main() -> None:
    validator = DocumentationValidator(STABLE_BASE, LATEST_BASE)

    # Step 1: Crawl stable documentation
    stable_urls = validator.crawl_stable_docs()

    # Step 2: Validate against latest documentation
    results = validator.validate_latest_docs(stable_urls)

    # Step 3: Print summary
    validator.print_summary(results)

    # Exit with error code if there are missing pages
    if results['missing'] or results['error']:
        exit(1)
    else:
        exit(0)

if __name__ == "__main__":
    main()
