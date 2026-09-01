"""Listener-supplied context: a document, a photo, or a link.

Three rules shape everything here.

**A failure must be loud.** If a PDF cannot be read, the listener is told so
before anything is generated. The alternative - attaching nothing and writing a
confident episode about a file it never saw - is the silent-success failure
this project has lost the most time to.

**Extraction happens once, on attach.** The text is pulled out and stored when
the file arrives, not when the episode is generated, so the cost lands while
someone is still typing rather than in front of the first word.

**Nothing here is shared.** An episode built on someone's document is theirs;
`pipeline` refuses to cache it and it never reaches Explore.
"""
from __future__ import annotations

import base64
import binascii
import io
import json
import logging
import re
import sqlite3
import threading
import time
import uuid
import zipfile
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

#: Biggest file accepted, before base64. Large enough for a long report, small
#: enough that a phone upload does not stall.
MAX_BYTES = 8 * 1024 * 1024
#: Extracted text kept per attachment. A whole book would swamp the prompt and
#: cost more than the episode is worth; this is several thousand words.
MAX_TEXT_CHARS = 24_000
#: How long an attachment stays usable. It is context for a search happening
#: now, not a document store.
TTL_SECONDS = 6 * 60 * 60

TEXT_SUFFIXES = (".txt", ".md", ".markdown", ".csv", ".tsv", ".json", ".log", ".rtf")
IMAGE_TYPES = {
    ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
    ".gif": "image/gif", ".webp": "image/webp",
}

KINDS = ("document", "image", "link")


class AttachmentError(Exception):
    """Something the listener needs to be told, in words they can act on."""


@dataclass
class Attachment:
    id: str
    kind: str
    name: str
    #: Extracted text. Empty for an image, which travels as pixels.
    text: str = ""
    #: Base64 payload and media type, images only.
    data_b64: str = ""
    media_type: str = ""
    url: str = ""
    created: float = field(default_factory=time.time)

    @property
    def chars(self) -> int:
        return len(self.text)

    def as_dict(self) -> dict:
        """What the interface needs. Never the payload - it already has it."""
        return {
            "id": self.id, "kind": self.kind, "name": self.name,
            "chars": self.chars, "url": self.url,
            "preview": " ".join(self.text.split())[:160],
        }

    def as_prompt_block(self) -> str:
        """How this appears to the script model. Images are not included here."""
        if self.kind == "link":
            return f'<attached kind="link" url="{self.url}" title="{self.name}">\n{self.text}\n</attached>'
        return f'<attached kind="document" name="{self.name}">\n{self.text}\n</attached>'


# --------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------

def _suffix(name: str) -> str:
    name = (name or "").strip().lower()
    return name[name.rfind("."):] if "." in name else ""


def _clip(text: str) -> str:
    text = re.sub(r"[ \t]+", " ", (text or "")).strip()
    if len(text) > MAX_TEXT_CHARS:
        # Say so in the text itself: the model must not describe a truncated
        # document as though it had read all of it.
        text = text[:MAX_TEXT_CHARS] + "\n\n[This document was longer and has been cut off here.]"
    return text


def _docx_text(data: bytes) -> str:
    """A .docx is a zip of XML. No dependency needed to read the words out."""
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as archive:
            xml = archive.read("word/document.xml").decode("utf-8", "replace")
    except (zipfile.BadZipFile, KeyError) as exc:
        raise AttachmentError("That .docx could not be opened.") from exc
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:tab[^>]*/>", "\t", xml)
    return re.sub(r"<[^>]+>", "", xml)


def _pdf_text(data: bytes) -> str:
    try:
        import pypdf
    except BaseException as exc:
        # BaseException, not Exception, on purpose. A missing package raises
        # ImportError, but pypdf's optional native crypto dependency can raise
        # pyo3's PanicException, which does not inherit from Exception - so a
        # narrower catch lets a broken install take down the request instead of
        # telling the listener their PDF cannot be read here.
        raise AttachmentError(
            "PDFs cannot be read on this machine (pypdf is missing or broken). "
            "Paste the text instead, or install it with: pip install pypdf"
        ) from exc
    try:
        reader = pypdf.PdfReader(io.BytesIO(data))
        pages = [page.extract_text() or "" for page in reader.pages[:80]]
    except Exception as exc:
        raise AttachmentError("That PDF could not be read - it may be scanned images.") from exc
    text = "\n\n".join(p for p in pages if p.strip())
    if not text.strip():
        raise AttachmentError(
            "That PDF has no text in it - it is probably a scan. A photo of the "
            "page would work better."
        )
    return text


def extract_document(name: str, data: bytes) -> str:
    """Words out of a file, or an error the listener can act on."""
    if len(data) > MAX_BYTES:
        raise AttachmentError(f"That file is over {MAX_BYTES // (1024 * 1024)} MB.")
    suffix = _suffix(name)
    if suffix == ".pdf":
        text = _pdf_text(data)
    elif suffix == ".docx":
        text = _docx_text(data)
    elif suffix in TEXT_SUFFIXES or not suffix:
        text = data.decode("utf-8", "replace")
    else:
        raise AttachmentError(
            f"{suffix or 'That file type'} is not supported. Try a PDF, a Word "
            "document, a text file, or a photo."
        )
    text = _clip(text)
    if not text:
        raise AttachmentError("There was no text in that file.")
    return text


_SCRIPT_OR_STYLE = re.compile(r"<(script|style)\b.*?</\1>", re.S | re.I)
_TAG = re.compile(r"<[^>]+>")
_TITLE = re.compile(r"<title[^>]*>(.*?)</title>", re.S | re.I)


def html_to_text(html: str) -> tuple[str, str]:
    """Title and readable text. Deliberately crude - no parser dependency."""
    title = ""
    found = _TITLE.search(html)
    if found:
        title = re.sub(r"\s+", " ", _TAG.sub("", found.group(1))).strip()
    body = _SCRIPT_OR_STYLE.sub(" ", html)
    body = re.sub(r"<(br|/p|/div|/li|/h[1-6])\s*/?>", "\n", body, flags=re.I)
    body = _TAG.sub(" ", body)
    for entity, char in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"),
                         ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'")):
        body = body.replace(entity, char)
    body = re.sub(r"\n{3,}", "\n\n", body)
    return title, _clip(body)


def fetch_link(url: str, timeout: float = 8.0) -> tuple[str, str]:
    """Read a page. Returns (title, text).

    This is a network round-trip before any script is written, which is exactly
    the cost the project refuses to pay per search. It is acceptable here only
    because it happens when the link is *attached* - while the listener is
    still typing - and never on the generation path.
    """
    import httpx

    if not re.match(r"^https?://", url or "", re.I):
        raise AttachmentError("A link needs to start with http:// or https://")
    try:
        with httpx.Client(follow_redirects=True, timeout=timeout) as client:
            response = client.get(url, headers={"User-Agent": "FAM/1.0 (+briefing)"})
            response.raise_for_status()
            body = response.text[: MAX_BYTES]
    except Exception as exc:
        raise AttachmentError(f"That link could not be opened ({type(exc).__name__}).") from exc
    title, text = html_to_text(body)
    if not text:
        raise AttachmentError("There was no readable text at that link.")
    return title or url, text


def build(kind: str, name: str = "", data_b64: str = "", url: str = "") -> Attachment:
    """Turn what the interface sent into a stored, extracted attachment."""
    if kind not in KINDS:
        raise AttachmentError("Unknown attachment type.")
    ident = uuid.uuid4().hex[:16]

    if kind == "link":
        title, text = fetch_link(url)
        return Attachment(id=ident, kind="link", name=title, text=text, url=url)

    try:
        data = base64.b64decode(data_b64 or "", validate=True)
    except (binascii.Error, ValueError) as exc:
        raise AttachmentError("That file did not arrive intact.") from exc
    if not data:
        raise AttachmentError("That file was empty.")
    if len(data) > MAX_BYTES:
        raise AttachmentError(f"That file is over {MAX_BYTES // (1024 * 1024)} MB.")

    if kind == "image":
        media = IMAGE_TYPES.get(_suffix(name))
        if not media:
            raise AttachmentError("Photos must be PNG, JPEG, GIF or WebP.")
        return Attachment(id=ident, kind="image", name=name or "photo",
                          data_b64=base64.b64encode(data).decode("ascii"),
                          media_type=media)

    return Attachment(id=ident, kind="document", name=name or "document",
                      text=extract_document(name, data))


# --------------------------------------------------------------------------
# Storage
# --------------------------------------------------------------------------

class AttachmentStore:
    """Short-lived, per-listener. Not a document store - context for a search."""

    def __init__(self, path: str = "attachments.db") -> None:
        self.path = path
        self._local = threading.local()
        with self._conn() as conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS attachments (
                       id         TEXT PRIMARY KEY,
                       user_id    TEXT NOT NULL,
                       kind       TEXT NOT NULL,
                       name       TEXT NOT NULL,
                       body       TEXT NOT NULL,
                       created    REAL NOT NULL
                   )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS attach_user ON attachments(user_id)")

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self.path, check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn = conn
        return conn

    def put(self, user_id: str, item: Attachment) -> Attachment:
        body = json.dumps({
            "text": item.text, "data_b64": item.data_b64,
            "media_type": item.media_type, "url": item.url,
        })
        with self._conn() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO attachments VALUES (?,?,?,?,?,?)",
                (item.id, user_id or "", item.kind, item.name, body, item.created),
            )
        return item

    def get(self, user_id: str, ident: str) -> Attachment | None:
        row = self._conn().execute(
            "SELECT id, kind, name, body, created FROM attachments"
            " WHERE id = ? AND user_id = ? AND created >= ?",
            (ident, user_id or "", time.time() - TTL_SECONDS),
        ).fetchone()
        if not row:
            return None
        body = json.loads(row[3])
        return Attachment(id=row[0], kind=row[1], name=row[2], created=row[4],
                          text=body.get("text", ""), data_b64=body.get("data_b64", ""),
                          media_type=body.get("media_type", ""), url=body.get("url", ""))

    def resolve(self, user_id: str, ids) -> list[Attachment]:
        """Every attachment that is still valid, in the order asked for."""
        out = []
        for ident in ids:
            item = self.get(user_id, (ident or "").strip())
            if item:
                out.append(item)
        return out

    def delete(self, user_id: str, ident: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM attachments WHERE id = ? AND user_id = ?",
                               (ident, user_id or ""))
        return bool(cur.rowcount)

    def purge_expired(self) -> int:
        try:
            with self._conn() as conn:
                cur = conn.execute("DELETE FROM attachments WHERE created < ?",
                                   (time.time() - TTL_SECONDS,))
            return cur.rowcount or 0
        except Exception:
            return 0
