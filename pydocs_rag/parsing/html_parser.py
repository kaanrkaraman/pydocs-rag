"""HTML parser for extracting structured content from Python documentation."""

import hashlib
import re
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString, Tag

from pydocs_rag.config import ParsingConfig
from pydocs_rag.ingestion.manifest import ManifestEntry
from pydocs_rag.models import DocumentMetadata, ParsedDocument, ParsedSection


class HTMLParser:
    def __init__(self, config: ParsingConfig) -> None:
        self.config = config
        self.nav_selectors = [
            "nav",
            ".navigation",
            ".nav",
            ".sphinxsidebar",
            ".related",
            "#navbar",
            ".navbar",
            "[role='navigation']",
        ]
        self.footer_selectors = [
            "footer",
            ".footer",
            "#footer",
            ".sphinxfooter",
            "[role='contentinfo']",
        ]
        self.sidebar_selectors = [
            "aside",
            ".sidebar",
            "#sidebar",
            ".sphinxsidebarwrapper",
            ".toctree-wrapper",
        ]

    def _remove_elements(self, soup: BeautifulSoup, selectors: list[str]) -> None:
        for selector in selectors:
            for element in soup.select(selector):
                element.decompose()

    def _extract_code_blocks(self, element: Tag) -> list[str]:
        code_blocks = []
        for pre in element.find_all("pre"):
            if code_text := pre.get_text(strip=True):
                code_blocks.append(code_text)
        for code in element.find_all("code"):
            if code.parent is None or code.parent.name != "pre":
                continue
            if (code_text := code.get_text(strip=True)) and code_text not in code_blocks:
                code_blocks.append(code_text)
        return code_blocks

    def _get_section_anchor(self, element: Tag) -> str | None:
        if elem_id := element.get("id"):
            return str(elem_id) if not isinstance(elem_id, list) else elem_id[0]
        if parent := element.find_parent(["section", "div"]):
            if parent_id := parent.get("id"):
                return str(parent_id) if not isinstance(parent_id, list) else parent_id[0]
        if (a_tag := element.find("a", class_="headerlink")) and isinstance(a_tag, Tag):
            if (href := a_tag.get("href")) and isinstance(href, str) and href.startswith("#"):
                return href[1:]
        return None

    def _extract_text_with_code(self, element: Tag) -> str:
        result = []

        for child in element.children:
            if isinstance(child, NavigableString):
                text = str(child).strip()
                if text:
                    result.append(text)
            elif isinstance(child, Tag):
                if child.name == "pre":
                    if code_text := child.get_text(strip=True):
                        result.append(f"\n```python\n{code_text}\n```\n")
                elif child.name == "code":
                    result.append(f"`{child.get_text(strip=True)}`")
                elif child.name in ("table",):
                    if self.config.preserve_tables:
                        result.append(self._extract_table(child))
                elif child.name in ("ul", "ol"):
                    result.append(self._extract_list(child))
                elif child.name in ("p", "div", "span", "a", "strong", "em", "b", "i"):
                    result.append(self._extract_text_with_code(child))
                else:
                    text = child.get_text(strip=True)
                    if text:
                        result.append(text)

        return " ".join(filter(None, result))

    def _extract_table(self, table: Tag) -> str:
        rows = []
        for tr in table.find_all("tr"):
            cells = []
            for td in tr.find_all(["td", "th"]):
                cells.append(td.get_text(strip=True))
            if cells:
                rows.append(" | ".join(cells))
        return "\n" + "\n".join(rows) + "\n" if rows else ""

    def _extract_list(self, list_element: Tag) -> str:
        items = []
        for li in list_element.find_all("li", recursive=False):
            text = li.get_text(strip=True)
            if text:
                prefix = "- " if list_element.name == "ul" else "1. "
                items.append(f"{prefix}{text}")
        return "\n" + "\n".join(items) + "\n" if items else ""

    def _find_main_content(self, soup: BeautifulSoup) -> Tag | None:
        main_selectors = [
            "main",
            "[role='main']",
            ".document",
            ".body",
            "#content",
            ".content",
            "article",
            ".documentwrapper",
        ]
        for selector in main_selectors:
            element = soup.select_one(selector)
            if element:
                return element
        return soup.body if soup.body else soup.find()

    def _extract_sections(self, content: Tag, parent_path: list[str]) -> list[ParsedSection]:
        sections = []
        current_heading = ""
        current_level = 0
        current_content_parts = []
        current_code_blocks: list[str] = []
        current_anchor = None

        heading_pattern = re.compile(r"^h([1-6])$")

        for element in content.children:
            if not isinstance(element, Tag):
                continue

            heading_match = heading_pattern.match(element.name)
            if heading_match:
                if current_heading or current_content_parts:
                    content_text = " ".join(current_content_parts).strip()
                    if content_text or current_code_blocks:
                        sections.append(
                            ParsedSection(
                                heading=current_heading,
                                heading_level=current_level,
                                content=content_text,
                                code_blocks=current_code_blocks,
                                anchor=current_anchor,
                                parent_path=parent_path.copy(),
                            )
                        )

                current_level = int(heading_match.group(1))
                current_heading = element.get_text(strip=True)
                current_heading = re.sub(r"\s*¶\s*$", "", current_heading)
                current_content_parts = []
                current_code_blocks = []
                current_anchor = self._get_section_anchor(element)

            elif element.name == "section":
                new_path = parent_path + [current_heading] if current_heading else parent_path
                nested_sections = self._extract_sections(element, new_path)
                sections.extend(nested_sections)

            elif element.name == "pre":
                if self.config.preserve_code_blocks:
                    code_text = element.get_text(strip=True)
                    if code_text:
                        current_code_blocks.append(code_text)
                        current_content_parts.append(f"\n```python\n{code_text}\n```\n")

            elif element.name in ("p", "div", "ul", "ol", "table", "dl"):
                text = self._extract_text_with_code(element)
                if len(text) >= self.config.min_paragraph_length:
                    current_content_parts.append(text)
                    current_code_blocks.extend(self._extract_code_blocks(element))

        if current_heading or current_content_parts:
            content_text = " ".join(current_content_parts).strip()
            if content_text or current_code_blocks:
                sections.append(
                    ParsedSection(
                        heading=current_heading,
                        heading_level=current_level,
                        content=content_text,
                        code_blocks=current_code_blocks,
                        anchor=current_anchor,
                        parent_path=parent_path.copy(),
                    )
                )

        return sections

    def parse_html(self, html_content: str, manifest_entry: ManifestEntry) -> ParsedDocument:
        soup = BeautifulSoup(html_content, "lxml")

        if self.config.remove_nav:
            self._remove_elements(soup, self.nav_selectors)
        if self.config.remove_footer:
            self._remove_elements(soup, self.footer_selectors)
        if self.config.remove_sidebar:
            self._remove_elements(soup, self.sidebar_selectors)

        title = manifest_entry.title
        if not title:
            title_tag = soup.find("title")
            if title_tag:
                title = title_tag.get_text(strip=True)
            else:
                h1_tag = soup.find("h1")
                if h1_tag:
                    title = h1_tag.get_text(strip=True)
                else:
                    title = "Untitled"

        main_content = self._find_main_content(soup)
        if main_content is not None:
            sections = self._extract_sections(main_content, [])
        else:
            sections = []

        doc_id = hashlib.sha256(manifest_entry.url.encode()).hexdigest()[:12]

        metadata = DocumentMetadata(
            url=manifest_entry.url,
            title=title,
            source_name=Path(manifest_entry.file_path).parts[0] if manifest_entry.file_path else "",
            crawl_timestamp=manifest_entry.crawl_timestamp,
            last_modified=manifest_entry.last_modified,
            etag=manifest_entry.etag,
            content_hash=manifest_entry.content_hash,
        )

        raw_text = soup.get_text(separator=" ", strip=True)

        return ParsedDocument(
            doc_id=doc_id,
            url=manifest_entry.url,
            title=title,
            sections=sections,
            metadata=metadata,
            raw_text=raw_text,
        )

    def parse_file(self, file_path: Path, manifest_entry: ManifestEntry) -> ParsedDocument:
        with open(file_path, encoding="utf-8", errors="replace") as f:
            html_content = f.read()
        return self.parse_html(html_content, manifest_entry)
