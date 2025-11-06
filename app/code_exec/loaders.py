from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import io
import tempfile


def _detect_encoding(sample: bytes) -> str:
    try:
        import chardet  # type: ignore

        res = chardet.detect(sample)
        return res.get("encoding") or "utf-8"
    except Exception:
        return "utf-8"


def _detect_delimiter(sample_text: str) -> Optional[str]:
    import csv

    try:
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample_text)
        return dialect.delimiter
    except Exception:
        return None


def load_csv(path: str) -> Dict[str, Any]:
    """CSV loader with basic encoding and delimiter inference.

    Returns a dict containing a pandas DataFrame and metadata.
    """
    try:
        import pandas as pd  # type: ignore
    except ImportError as e:
        return {"type": "error", "source": path, "error": "pandas not installed"}

    if not os.path.exists(path):
        return {"type": "error", "source": path, "error": "file not found"}

    with open(path, "rb") as f:
        head = f.read(100_000)
    encoding = _detect_encoding(head)
    sample_text = head.decode(encoding, errors="ignore")
    delimiter = _detect_delimiter(sample_text)

    read_kwargs: Dict[str, Any] = {"encoding": encoding, "low_memory": False}
    if delimiter:
        read_kwargs["sep"] = delimiter

    try:
        df = pd.read_csv(path, **read_kwargs)
        meta = {
            "loader": "csv",
            "encoding": encoding,
            "delimiter": delimiter or ",",
            "rows": int(df.shape[0]),
            "cols": int(df.shape[1]),
        }
        return {"type": "table", "source": path, "dataframe": df, "meta": meta}
    except Exception as e:
        return {"type": "error", "source": path, "error": str(e)}


def load_html(path_or_content: str) -> Dict[str, Any]:
    """HTML loader extracting tables and basic metadata.

    Supports local .html/.htm files. For remote content, use load_url later.
    Returns tables as list of pandas DataFrames and minimal page metadata.
    """
    try:
        import pandas as pd  # type: ignore
    except ImportError:
        return {"type": "error", "source": path_or_content, "error": "pandas not installed"}

    html_text: Optional[str] = None
    source = path_or_content
    if os.path.exists(path_or_content):
        try:
            # Attempt to detect encoding for HTML as well
            with open(path_or_content, "rb") as f:
                head = f.read()
            encoding = _detect_encoding(head)
            html_text = head.decode(encoding, errors="ignore")
        except Exception as e:
            return {"type": "error", "source": source, "error": str(e)}
    else:
        # Treat input as raw HTML string (fallback), not URL
        html_text = path_or_content

    tables: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"loader": "html", "source_kind": "file" if os.path.exists(source) else "string"}

    # Basic metadata via BeautifulSoup if available
    try:
        from bs4 import BeautifulSoup  # type: ignore

        soup = BeautifulSoup(html_text, "html.parser")
        title = soup.title.string.strip() if soup.title and soup.title.string else None
        meta_tags = { (m.get("name") or m.get("property") or f"meta_{i}"): m.get("content") for i, m in enumerate(soup.find_all("meta")) if m.get("content") }
        meta.update({"title": title, "meta": meta_tags})
    except Exception:
        # BeautifulSoup optional; ignore if missing
        pass

    # Extract tables using pandas.read_html
    try:
        dfs = pd.read_html(html_text) if html_text else []
        for idx, df in enumerate(dfs):
            tables.append({
                "type": "table",
                "source": source,
                "dataframe": df,
                "meta": {"index": idx, "rows": int(df.shape[0]), "cols": int(df.shape[1]), "origin": "html_table"},
            })
        return {"type": "mixed", "source": source, "tables": tables, "text": "", "meta": meta}
    except Exception as e:
        return {"type": "error", "source": source, "error": str(e)}


def load_xml(path: str) -> Dict[str, Any]:
    """XML loader that normalizes to records (list[dict]) and optional DataFrame.

    Uses xmltodict if available, falling back to lxml. Produces a flattened
    list of records where feasible.
    """
    if not os.path.exists(path):
        return {"type": "error", "source": path, "error": "file not found"}

    raw: Optional[str] = None
    try:
        with open(path, "rb") as f:
            raw_bytes = f.read()
        encoding = _detect_encoding(raw_bytes)
        raw = raw_bytes.decode(encoding, errors="ignore")
    except Exception as e:
        return {"type": "error", "source": path, "error": str(e)}

    records: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"loader": "xml"}

    def _flatten(obj: Any, prefix: str = "") -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        if isinstance(obj, dict):
            for k, v in obj.items():
                key = f"{prefix}.{k}" if prefix else str(k)
                out.update(_flatten(v, key))
        elif isinstance(obj, list):
            # Convert list to indexed keys
            for i, v in enumerate(obj):
                key = f"{prefix}[{i}]"
                out.update(_flatten(v, key))
        else:
            out[prefix or "value"] = obj
        return out

    try:
        try:
            import xmltodict  # type: ignore

            parsed = xmltodict.parse(raw)
            # Heuristic: collect leaf lists of dicts as records; otherwise flatten root
            def _collect_records(node: Any) -> Optional[List[Dict[str, Any]]]:
                if isinstance(node, list) and node and all(isinstance(x, dict) for x in node):
                    return [ _flatten(x) for x in node ]
                if isinstance(node, dict):
                    for v in node.values():
                        recs = _collect_records(v)
                        if recs is not None:
                            return recs
                return None

            recs = _collect_records(parsed)
            if recs is None:
                recs = [_flatten(parsed)]
            records = recs
            meta["parser"] = "xmltodict"
        except Exception:
            from lxml import etree  # type: ignore

            root = etree.fromstring(raw.encode("utf-8", errors="ignore"))
            # Fallback: create one record with tag counts
            tags: Dict[str, int] = {}
            for elem in root.iter():
                tags[elem.tag] = tags.get(elem.tag, 0) + 1
            records = [tags]
            meta["parser"] = "lxml"
    except Exception as e:
        return {"type": "error", "source": path, "error": str(e)}

    # Optional DataFrame construction when records look tabular
    df_meta: Dict[str, Any] = {}
    df_obj: Any = None
    try:
        import pandas as pd  # type: ignore

        if records and isinstance(records, list) and all(isinstance(r, dict) for r in records):
            df_obj = pd.DataFrame.from_records(records)
            df_meta = {"rows": int(df_obj.shape[0]), "cols": int(df_obj.shape[1])}
    except Exception:
        pass

    return {
        "type": "records",
        "source": path,
        "records": records,
        "dataframe": df_obj,
        "meta": {**meta, **df_meta},
    }


def load_pdf(path: str) -> Dict[str, Any]:
    """PDF loader extracting text and tables where possible.

    Prefers pdfplumber for text; tries camelot for tables if installed; falls back
    to returning only text when table extraction is unavailable.
    """
    if not os.path.exists(path):
        return {"type": "error", "source": path, "error": "file not found"}

    text_content = ""
    tables: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {"loader": "pdf"}

    # Text extraction
    try:
        import pdfplumber  # type: ignore

        with pdfplumber.open(path) as pdf:
            pages_text = []
            for page in pdf.pages:
                try:
                    pages_text.append(page.extract_text() or "")
                except Exception:
                    pages_text.append("")
            text_content = "\n".join(pages_text)
        meta["text_extractor"] = "pdfplumber"
    except Exception:
        # Optional: no text extracted if library missing
        meta["text_extractor"] = "none"

    # Table extraction (optional)
    try:
        import camelot  # type: ignore
        import pandas as pd  # type: ignore

        try:
            tables_extracted = camelot.read_pdf(path, pages="all")
            for idx, t in enumerate(tables_extracted):
                try:
                    df = t.df
                    if hasattr(df, "shape"):
                        tables.append({
                            "type": "table",
                            "source": path,
                            "dataframe": df,
                            "meta": {"index": idx, "rows": int(df.shape[0]), "cols": int(df.shape[1]), "origin": "pdf_table"},
                        })
                except Exception:
                    continue
            meta["table_extractor"] = "camelot"
        except Exception:
            meta["table_extractor"] = "none"
    except Exception:
        meta["table_extractor"] = "none"

    return {"type": "mixed", "source": path, "tables": tables, "text": text_content, "meta": meta}


def load_url(url: str) -> Dict[str, Any]:
    """URL loader that fetches content and routes to specific parsers by MIME/extension."""
    try:
        import requests  # type: ignore
        import pandas as pd  # type: ignore
    except Exception:
        return {"type": "error", "source": url, "error": "requests/pandas not installed"}

    try:
        resp = requests.get(url, timeout=20)
        ct = (resp.headers.get("Content-Type") or "").lower()
        status = resp.status_code
        if status != 200:
            return {"type": "error", "source": url, "error": f"http {status}"}
        content = resp.content
    except Exception as e:
        return {"type": "error", "source": url, "error": str(e)}

    # Route by content-type first
    if "text/csv" in ct or url.lower().endswith(".csv"):
        encoding = _detect_encoding(content)
        try:
            df = pd.read_csv(io.StringIO(content.decode(encoding, errors="ignore")))
            meta = {"loader": "url->csv", "encoding": encoding, "rows": int(df.shape[0]), "cols": int(df.shape[1])}
            return {"type": "table", "source": url, "dataframe": df, "meta": meta}
        except Exception as e:
            return {"type": "error", "source": url, "error": f"csv parse: {e}"}

    if "text/html" in ct or url.lower().endswith((".html", ".htm")):
        try:
            html_text = content.decode(_detect_encoding(content), errors="ignore")
            # Reuse our HTML loader path by passing raw string
            return load_html(html_text)
        except Exception as e:
            return {"type": "error", "source": url, "error": f"html parse: {e}"}

    if "application/xml" in ct or "text/xml" in ct or url.lower().endswith(".xml"):
        try:
            encoding = _detect_encoding(content)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".xml") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            res = load_xml(tmp_path)
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            return res
        except Exception as e:
            return {"type": "error", "source": url, "error": f"xml parse: {e}"}

    if "application/pdf" in ct or url.lower().endswith(".pdf"):
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            res = load_pdf(tmp_path)
            try:
                os.unlink(tmp_path)
            except Exception:
                pass
            return res
        except Exception as e:
            return {"type": "error", "source": url, "error": f"pdf parse: {e}"}

    # Fallback: treat as text blob
    try:
        text = content.decode(_detect_encoding(content), errors="ignore")
    except Exception:
        text = ""
    return {"type": "text", "source": url, "text": text, "meta": {"loader": "url", "content_type": ct}}


def normalize_inputs(files: Optional[List[str]] = None, urls: Optional[List[str]] = None) -> Dict[str, Any]:
    """Call specific loaders by extension and build canonical output."""
    tables: List[Dict[str, Any]] = []
    texts: List[Dict[str, Any]] = []

    for p in files or []:
        ext = os.path.splitext(p)[1].lower()
        if ext == ".csv":
            res = load_csv(p)
            if res.get("type") == "table":
                tables.append(res)
            else:
                texts.append({"type": "log", "message": f"CSV load issue: {res.get('error')}"})
        elif ext in (".html", ".htm"):
            res = load_html(p)
            if res.get("type") == "mixed":
                tables.extend(res.get("tables", []))
                texts.append({"type": "log", "message": "HTML parsed with tables extracted"})
            else:
                texts.append({"type": "log", "message": f"HTML load issue: {res.get('error')}"})
        elif ext == ".xml":
            res = load_xml(p)
            if res.get("type") == "records":
                if res.get("dataframe") is not None:
                    tables.append({
                        "type": "table",
                        "source": p,
                        "dataframe": res.get("dataframe"),
                        "meta": {"origin": "xml_records", **res.get("meta", {})},
                    })
                texts.append({"type": "log", "message": "XML parsed into records"})
            else:
                texts.append({"type": "log", "message": f"XML load issue: {res.get('error')}"})
        elif ext == ".pdf":
            res = load_pdf(p)
            if res.get("type") == "mixed":
                tables.extend(res.get("tables", []))
                if res.get("text"):
                    texts.append({"type": "text", "source": p, "text": res.get("text")})
            elif res.get("type") == "text":
                texts.append(res)
            else:
                texts.append({"type": "log", "message": f"PDF load issue: {res.get('error')}"})
        else:
            texts.append({"type": "log", "message": f"Unsupported file type: {ext}"})

    # URLs
    for u in urls or []:
        res = load_url(u)
        rtype = res.get("type")
        if rtype == "table":
            tables.append(res)
        elif rtype == "mixed":
            tables.extend(res.get("tables", []))
            if res.get("text"):
                texts.append({"type": "text", "source": u, "text": res.get("text")})
        elif rtype == "text":
            texts.append(res)
        else:
            texts.append({"type": "log", "message": f"URL load issue: {res.get('error')}"})

    meta = {"files": files or [], "urls": urls or []}
    return {"tables": tables, "texts": texts, "metadata": meta}


