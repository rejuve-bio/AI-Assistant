import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
import re
import os
import pymupdf4llm


def extract_markdown_from_pdf(pdf_path):
    absolute_path = os.path.abspath(pdf_path)
    markdown_text = pymupdf4llm.to_markdown(absolute_path)
    return markdown_text


def markdown_to_clean_text(markdown_text):
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
    markdown_text = re.sub(r"```[\s\S]*?```", "", markdown_text)  # Remove code blocks
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
    markdown_text = re.sub(r"[ \t]+", " ", markdown_text)  # Multiple spaces to single
    markdown_text = re.sub(
        r"\n[ \t]+", "\n", markdown_text
    )  # Remove leading spaces on lines
    return markdown_text.strip()


def chunk_by_sections(text):
    # Chunk text by document sections first, then by size if needed
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
                    sub_chunks = chunk_text_recursive(current_section)
                    chunks.extend(sub_chunks)
                else:
                    chunks.append(current_section.strip())
            current_section = section
        else:
            current_section += section
    if current_section.strip():
        if len(current_section) > 1000:
            sub_chunks = chunk_text_recursive(current_section)
            chunks.extend(sub_chunks)
        else:
            chunks.append(current_section.strip())
    return chunks


def chunk_text_recursive(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
    )
    return splitter.split_text(text)


def extract_text_from_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")
    markdown_text = extract_markdown_from_pdf(pdf_path)
    clean_text = markdown_to_clean_text(markdown_text)
    return clean_text


def chunk_text(text):
    if not text or not text.strip():
        return []
    return chunk_by_sections(text)
