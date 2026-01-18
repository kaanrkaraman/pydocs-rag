"""Web crawler for Python documentation sources."""

import asyncio
import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

from pydocs_rag.config import CorpusConfig, SourceConfig
from pydocs_rag.ingestion.manifest import Manifest, ManifestEntry
from pydocs_rag.version import compute_content_hash


class Crawler:
    def __init__(
        self,
        config: CorpusConfig,
        output_dir: Path,
        resume: bool = True,
        batch_size: int = 5,
    ) -> None:
        self.config = config
        self.output_dir = output_dir
        self.resume = resume
        self.batch_size = batch_size
        self.visited_urls: set[str] = set()
        self.robots_parsers: dict[str, RobotFileParser] = {}
        self.manifest = Manifest(sources=[s.url for s in config.sources])
        self._pending_links: list[str] = []

    def _normalize_url(self, url: str) -> str:
        parsed = urlparse(url)
        path = parsed.path.rstrip("/") or "/"
        return f"{parsed.scheme}://{parsed.netloc}{path}"

    def _url_to_filepath(self, url: str, source_name: str) -> Path:
        parsed = urlparse(url)
        if not (path := parsed.path.lstrip("/")):
            path = "index"
        path = re.sub(r"[^\w\-_./]", "_", path)
        if not path.endswith(".html"):
            path = f"{path}.html"
        return self.output_dir / source_name / path

    def _matches_patterns(self, url: str, patterns: list[str], base_url: str) -> bool:
        if not patterns:
            return True
        parsed = urlparse(url)
        path = parsed.path

        include_patterns = [p for p in patterns if not p.startswith("!")]
        exclude_patterns = [p[1:] for p in patterns if p.startswith("!")]

        for pattern in exclude_patterns:
            if pattern.startswith("/"):
                if path.startswith(pattern.rstrip("*")):
                    return False
            elif pattern in path:
                return False

        if not include_patterns:
            return True

        for pattern in include_patterns:
            if pattern.startswith("/"):
                if path.startswith(pattern.rstrip("*")):
                    return True
            elif pattern in path:
                return True
        return False

    async def _get_robots_parser(
        self, base_url: str, client: httpx.AsyncClient
    ) -> tuple[RobotFileParser, bool]:
        if base_url in self.robots_parsers:
            return self.robots_parsers[base_url]

        parser = RobotFileParser()
        allow_all = False
        robots_url = urljoin(base_url, "/robots.txt")
        try:
            response = await client.get(robots_url)
            if response.status_code == 200:
                parser.parse(response.text.splitlines())
            else:
                allow_all = True
        except Exception:
            allow_all = True
        self.robots_parsers[base_url] = (parser, allow_all)
        return parser, allow_all

    def _can_fetch(self, url: str, parser: RobotFileParser, allow_all: bool) -> bool:
        if not self.config.respect_robots_txt or allow_all:
            return True
        return parser.can_fetch(self.config.user_agent, url)

    def _extract_links(self, html: str, base_url: str, source_config: SourceConfig) -> list[str]:
        soup = BeautifulSoup(html, "lxml")
        links = []
        base_domain = urlparse(source_config.url).netloc

        for a_tag in soup.find_all("a", href=True):
            href_attr = a_tag["href"]
            href = str(href_attr) if not isinstance(href_attr, list) else href_attr[0]
            if href.startswith("#") or href.startswith("javascript:"):
                continue

            full_url = urljoin(base_url, href)
            parsed = urlparse(full_url)

            if parsed.netloc != base_domain:
                continue
            if parsed.scheme not in ("http", "https"):
                continue

            if (normalized := self._normalize_url(full_url)) not in self.visited_urls:
                if self._matches_patterns(
                    normalized, source_config.include_patterns, source_config.url
                ):
                    links.append(normalized)

        return links

    def _extract_title(self, html: str) -> str:
        soup = BeautifulSoup(html, "lxml")
        if title_tag := soup.find("title"):
            return title_tag.get_text(strip=True)
        if h1_tag := soup.find("h1"):
            return h1_tag.get_text(strip=True)
        return ""

    async def _crawl_url(
        self,
        url: str,
        source_config: SourceConfig,
        client: httpx.AsyncClient,
        robots_parser: RobotFileParser,
        allow_all: bool = False,
    ) -> tuple[ManifestEntry | None, list[str]]:
        if url in self.visited_urls:
            return None, []

        if not self._can_fetch(url, robots_parser, allow_all):
            return None, []

        self.visited_urls.add(url)
        file_path = self._url_to_filepath(url, source_config.name)

        if self.resume and file_path.exists():
            if existing_entry := self.manifest.get_entry_by_url(url):
                try:
                    with open(file_path, "rb") as f:
                        html_content = f.read()
                    links = self._extract_links(
                        html_content.decode("utf-8", errors="replace"),
                        url,
                        source_config,
                    )
                    return existing_entry, links
                except Exception:
                    pass

        try:
            response = await client.get(url, follow_redirects=True)
            crawl_timestamp = datetime.now()

            if response.status_code != 200:
                entry = ManifestEntry(
                    url=url,
                    crawl_timestamp=crawl_timestamp,
                    http_status=response.status_code,
                    content_hash="",
                    file_path="",
                )
                return entry, []

            html_content = response.content
            content_hash = compute_content_hash(html_content)

            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(html_content)

            html_text = html_content.decode("utf-8", errors="replace")
            title = self._extract_title(html_text)

            entry = ManifestEntry(
                url=url,
                crawl_timestamp=crawl_timestamp,
                http_status=response.status_code,
                content_hash=content_hash,
                file_path=str(file_path.relative_to(self.output_dir)),
                last_modified=response.headers.get("Last-Modified"),
                etag=response.headers.get("ETag"),
                content_length=len(html_content),
                title=title,
            )

            links = self._extract_links(html_text, url, source_config)
            return entry, links

        except Exception as e:
            entry = ManifestEntry(
                url=url,
                crawl_timestamp=datetime.now(),
                http_status=0,
                content_hash="",
                file_path="",
                title=f"Error: {str(e)[:100]}",
            )
            return entry, []

    async def _crawl_batch(
        self,
        urls: list[str],
        source_config: SourceConfig,
        client: httpx.AsyncClient,
        robots_parser: RobotFileParser,
        allow_all: bool = False,
    ) -> list[tuple[ManifestEntry | None, list[str]]]:
        async with asyncio.TaskGroup() as tg:
            tasks = [
                tg.create_task(
                    self._crawl_url(url, source_config, client, robots_parser, allow_all)
                )
                for url in urls
            ]
        return [task.result() for task in tasks]

    async def _crawl_source(self, source_config: SourceConfig, client: httpx.AsyncClient) -> int:
        robots_parser, allow_all = await self._get_robots_parser(source_config.url, client)
        source_domain = urlparse(source_config.url).netloc

        existing_source_urls = {
            e.url
            for e in self.manifest.entries
            if urlparse(e.url).netloc == source_domain and e.http_status == 200
        }
        crawled_count = len(existing_source_urls)

        if crawled_count >= self.config.max_pages_per_source:
            print(f"  Already at limit ({crawled_count} pages)")
            return 0

        failed_source_urls = {
            e.url
            for e in self.manifest.entries
            if urlparse(e.url).netloc == source_domain and e.http_status != 200
        }
        self.visited_urls -= failed_source_urls

        queue = []
        if self.resume and existing_source_urls:
            for entry in self.manifest.entries:
                if entry.url in existing_source_urls and entry.file_path:
                    file_path = self.output_dir / entry.file_path
                    if file_path.exists():
                        try:
                            with open(file_path, "rb") as f:
                                html = f.read().decode("utf-8", errors="replace")
                            links = self._extract_links(html, entry.url, source_config)
                            for link in links:
                                if link not in self.visited_urls and link not in queue:
                                    queue.append(link)
                        except Exception:
                            pass
            print(f"  Resuming with {crawled_count} existing, {len(queue)} links in queue")
        else:
            start_url = self._normalize_url(source_config.url)
            self.visited_urls.discard(start_url)
            queue = [start_url]

        new_crawled = 0
        while queue and crawled_count < self.config.max_pages_per_source:
            batch = []
            while queue and len(batch) < self.batch_size:
                url = queue.pop(0)
                if url not in self.visited_urls:
                    batch.append(url)

            if not batch:
                break

            try:
                results = await self._crawl_batch(
                    batch, source_config, client, robots_parser, allow_all
                )
            except ExceptionGroup as eg:
                print(f"  Batch errors: {len(eg.exceptions)} failures")
                results = []

            for entry, new_links in results:
                if entry:
                    self.manifest.add_entry(entry)
                    if entry.http_status == 200:
                        crawled_count += 1
                        new_crawled += 1
                        print(f"  [{new_crawled}] {entry.url[:80]}")

                for link in new_links:
                    if link not in self.visited_urls and link not in queue:
                        queue.append(link)

            await asyncio.sleep(self.config.crawl_delay_seconds)

        return new_crawled

    async def crawl(self) -> Manifest:
        print(
            f"Starting crawl with {len(self.config.sources)} sources (batch_size={self.batch_size})"
        )
        print(f"Output directory: {self.output_dir}")

        manifest_path = self.output_dir / "manifest.json"
        if self.resume and manifest_path.exists():
            self.manifest = Manifest.load(manifest_path)
            self.visited_urls = {e.url for e in self.manifest.entries}
            print(f"Resumed from existing manifest with {len(self.visited_urls)} URLs")

        async with httpx.AsyncClient(
            timeout=30.0,
            headers={"User-Agent": self.config.user_agent},
            follow_redirects=True,
        ) as client:
            for source_config in self.config.sources:
                print(f"\nCrawling source: {source_config.name} ({source_config.url})")
                count = await self._crawl_source(source_config, client)
                print(f"  Crawled {count} pages from {source_config.name}")

        self.manifest.save(manifest_path)
        print(f"\nCrawl complete. Corpus version: {self.manifest.corpus_version}")
        print(f"Total pages: {len(self.manifest.entries)}")
        print(f"Successful: {len(self.manifest.get_successful_entries())}")

        return self.manifest


def run_crawler(config: CorpusConfig, output_dir: Path, resume: bool = True) -> Manifest:
    crawler = Crawler(config, output_dir, resume)
    return asyncio.run(crawler.crawl())
