import fitz  # PyMuPDF
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
import trafilatura
import requests
from bs4 import BeautifulSoup
import logging
from urllib.parse import urlparse
import os
import re
import pymupdf4llm


logger = logging.getLogger(__name__)


class ContentProcessor:
    # Unified content processor for PDF, web, and generic text processing.
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
        )

    # PDF Processing Methods
    def extract_text_from_pdf(self, pdf_path):
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        markdown_text = self.extract_markdown_from_pdf(pdf_path)
        clean_text = self.markdown_to_clean_text(markdown_text)
        return clean_text

    def extract_markdown_from_pdf(self, pdf_path):
        absolute_path = os.path.abspath(pdf_path)
        return pymupdf4llm.to_markdown(absolute_path)

    def markdown_to_clean_text(self, markdown_text):
        markdown_text = re.sub(
            r"^#{1}\s+(.+)$", r"\1:", markdown_text, flags=re.MULTILINE
        )  # H1
        markdown_text = re.sub(
            r"^#{2}\s+(.+)$", r"\1:", markdown_text, flags=re.MULTILINE
        )  # H2
        markdown_text = re.sub(
            r"^#{3}\s+(.+)$", r"\1:", markdown_text, flags=re.MULTILINE
        )  # H3
        markdown_text = re.sub(
            r"^#{4,6}\s+(.+)$", r"\1:", markdown_text, flags=re.MULTILINE
        )  # H4-H6
        markdown_text = re.sub(r"\*\*(.+?)\*\*", r"\1", markdown_text)  # Remove bold
        markdown_text = re.sub(r"\*(.+?)\*", r"\1", markdown_text)  # Remove italic
        markdown_text = re.sub(
            r"__(.+?)__", r"\1", markdown_text
        )  # Remove bold (diff. syntax)
        markdown_text = re.sub(
            r"_(.+?)_", r"\1", markdown_text
        )  # Remove italic (diff. syntax)
        markdown_text = re.sub(
            r"^\s*[-*+]\s+", r"• ", markdown_text, flags=re.MULTILINE
        )  # Convert lists to bullet points
        markdown_text = re.sub(
            r"^\s*\d+\.\s+", r"• ", markdown_text, flags=re.MULTILINE
        )  # Remove numbered lists to bullet points
        markdown_text = re.sub(
            r"^\s*>\s+", r"", markdown_text, flags=re.MULTILINE
        )  # Remove blockquotes
        markdown_text = re.sub(
            r"```[\s\S]*?```", "", markdown_text
        )  # Remove code blocks
        markdown_text = re.sub(r"`([^`]+)`", r"\1", markdown_text)  # Remove inline code
        markdown_text = re.sub(
            r"^[-*_]{3,}$", "", markdown_text, flags=re.MULTILINE
        )  # Remove horizontal rules
        markdown_text = re.sub(
            r"\[([^\]]+)\]\([^)]+\)", r"\1", markdown_text
        )  # Remove links
        markdown_text = re.sub(
            r"\n\s*\n\s*\n", r"\n\n", markdown_text
        )  # Multiple newlines to double
        markdown_text = re.sub(
            r"[ \t]+", " ", markdown_text
        )  # Multiple spaces to single
        markdown_text = re.sub(
            r"\n[ \t]+", "\n", markdown_text
        )  # Remove leading spaces on lines
        return markdown_text.strip()

    # Web Processing Methods
    def extract_text_from_url(self, url):
        logger.info(f"Starting content extraction from {url}")

        if not url or not url.strip():
            logger.error("Empty URL provided")
            return None

        url = url.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url

        result = self.extract_text_with_trafilatura(url)

        if result is None or len(result["text"]) < 500:
            logger.info(f"Trafilatura failed for {url}, trying BeautifulSoup4")
            result = self.extract_text_with_bs4(url)

        if result is None:
            logger.error(f"Both extraction methods failed for {url}")
            return None

        logger.info(f"Successfully extracted content from {url}")
        return result

    def extract_text_with_trafilatura(self, url):
        try:
            logger.info(f"Attempting Trafilatura extraction for {url}")
            downloaded = trafilatura.fetch_url(url)
            if downloaded is None:
                logger.warning(f"Trafilatura failed to fetch {url}")
                return None

            text = trafilatura.extract(
                downloaded,
                include_comments=False,
                include_tables=False,
                favor_precision=True,
            )

            if not text or len(text.strip()) < 100:
                logger.warning(f"Trafilatura extracted too little content from {url}")
                return None

            logger.info(
                f"Trafilatura successfully extracted {len(text)} characters from {url}"
            )

            try:
                metadata = trafilatura.extract_metadata(downloaded)
                title = metadata.get("title") if metadata else None
                author = metadata.get("author") if metadata else None
                date = metadata.get("date") if metadata else None
            except Exception as e:
                logger.warning(f"Failed to extract metadata from {url}: {e}")
                title = None
                author = None
                date = None

            return {
                "text": text,
                "metadata": {
                    "title": title,
                    "author": author,
                    "date": date,
                    "url": url,
                },
            }
        except Exception as e:
            logger.error(f"Trafilatura extraction failed for {url}: {e}")
            return None

    def extract_text_with_bs4(self, url):
        try:
            logger.info(f"Attempting BeautifulSoup4 extraction for {url}")
            response = self.session.get(url, timeout=30)
            response.raise_for_status()

            soup = BeautifulSoup(response.content, "html.parser")

            for script in soup(["script", "style", "nav", "header", "footer", "aside"]):
                script.decompose()

            main_content = None
            selectors = [
                "main",
                "article",
                ".content",
                ".post",
                ".entry",
                ".article-content",
                ".post-content",
                ".entry-content",
                "#content",
                "#main",
                "#article",
            ]

            for selector in selectors:
                main_content = soup.select_one(selector)
                if main_content:
                    break

            if not main_content:
                main_content = soup.body

            if not main_content:
                logger.warning(f"No content found in {url}")
                return None

            text = main_content.get_text(separator="\n", strip=True)
            text = re.sub(r"\n\s*\n", "\n\n", text)
            text = re.sub(r"\s+", " ", text)

            if len(text.strip()) < 100:
                logger.warning(
                    f"BeautifulSoup4 extracted too little content from {url}"
                )
                return None

            title = soup.title.string if soup.title else None
            author = None

            author_selectors = [
                '[name="author"]',
                ".author",
                ".byline",
                '[rel="author"]',
                ".post-author",
                ".entry-author",
            ]

            for selector in author_selectors:
                author_elem = soup.select_one(selector)
                if author_elem:
                    author = author_elem.get_text(strip=True)
                    break

            logger.info(
                f"BeautifulSoup4 successfully extracted {len(text)} characters from {url}"
            )

            return {
                "text": text,
                "metadata": {"title": title, "author": author, "url": url},
            }
        except Exception as e:
            logger.error(f"BeautifulSoup4 extraction failed for {url}: {e}")
            return None

    def validate_url(self, url):
        try:
            parsed = urlparse(url)
            if not parsed.scheme or not parsed.netloc:
                return False, "Invalid URL format"

            response = self.session.head(url, timeout=10)
            if response.status_code != 200:
                return False, f"URL returned status code {response.status_code}"

            return True, "URL is valid and accessible"
        except Exception as e:
            return False, f"URL validation failed: {str(e)}"

    def clean_text_content(self, text):
        if not text:
            return ""

        text = re.sub(r"\s+", " ", text)
        text = re.sub(
            r"cookie|privacy|terms|advertisement", "", text, flags=re.IGNORECASE
        )
        text = re.sub(r"\n\s*\n", "\n\n", text)

        return text.strip()

    def chunk_text(self, text):
        if not text or not text.strip():
            return []
        return self.chunk_by_sections(text)

    def chunk_by_sections(self, text):
        chunks = []
        section_patterns = [
            r"^[A-Z][A-Za-z\s]+:",
            r"^\d+\.\s+[A-Z]",
            r"^[A-Z][A-Za-z\s]+\s+\d+",
        ]
        combined_pattern = "|".join(section_patterns)
        sections = re.split(f"({combined_pattern})", text, flags=re.MULTILINE)
        current_section = ""

        for i, section in enumerate(sections):
            if re.match(combined_pattern, section):
                if current_section.strip():
                    if len(current_section) > 1000:
                        sub_chunks = self.chunk_text_recursive(current_section)
                        chunks.extend(sub_chunks)
                    else:
                        chunks.append(current_section.strip())
                current_section = section
            else:
                current_section += section

        if current_section.strip():
            if len(current_section) > 1000:
                sub_chunks = self.chunk_text_recursive(current_section)
                chunks.extend(sub_chunks)
            else:
                chunks.append(current_section.strip())

        return chunks

    def chunk_text_recursive(self, text):
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
        )
        return splitter.split_text(text)

    # Convenience Methods
    def process_pdf(self, pdf_path):
        text = self.extract_text_from_pdf(pdf_path)
        return self.chunk_text(text)

    def process_web_content(self, url):
        result = self.extract_text_from_url(url)
        if not result:
            return None

        text = result["text"]
        cleaned_text = self.clean_text_content(text)
        chunks = self.chunk_text(cleaned_text)

        return {"chunks": chunks, "metadata": result["metadata"]}
