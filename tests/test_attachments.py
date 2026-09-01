"""Listener-supplied context: documents, photos and links.

The theme of these tests is that a failure must be visible. An episode written
about a document that was never actually read is worse than an error message,
and it is the failure this project has lost the most time to.
"""
from __future__ import annotations

import base64
import io
import os
import sys
import time
import zipfile

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import attachments as A  # noqa: E402
from script_generator import build_prompt, plan_episode  # noqa: E402


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def docx_bytes(text: str) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr(
            "word/document.xml",
            f"<w:document><w:body><w:p><w:r><w:t>{text}</w:t></w:r></w:p></w:body></w:document>",
        )
    return buf.getvalue()


# --- extraction -----------------------------------------------------------

def test_plain_text_is_read_as_is():
    assert A.extract_document("notes.txt", b"Revenue fell 12 percent.") == "Revenue fell 12 percent."


def test_a_docx_is_read_without_a_dependency():
    """A .docx is a zip of XML, so this needs no library and cannot rot."""
    text = A.extract_document("report.docx", docx_bytes("The margin held at 41 percent."))
    assert "margin held at 41 percent" in text


def test_a_corrupt_docx_says_so():
    with pytest.raises(A.AttachmentError, match="could not be opened"):
        A.extract_document("report.docx", b"not a zip")


def test_an_unsupported_type_names_what_would_work():
    with pytest.raises(A.AttachmentError) as caught:
        A.extract_document("thing.exe", b"MZ")
    assert "PDF" in str(caught.value)


def test_an_empty_file_is_an_error_not_an_empty_episode():
    with pytest.raises(A.AttachmentError):
        A.extract_document("empty.txt", b"")


def test_an_oversized_file_is_refused():
    with pytest.raises(A.AttachmentError, match="over"):
        A.extract_document("big.txt", b"x" * (A.MAX_BYTES + 1))


def test_a_long_document_says_in_its_own_text_that_it_was_cut():
    """The model must not describe a truncated document as if it read it all."""
    text = A.extract_document("long.txt", b"word " * (A.MAX_TEXT_CHARS))
    assert len(text) < A.MAX_TEXT_CHARS + 200
    assert "cut off" in text


def test_html_is_reduced_to_readable_text():
    title, text = A.html_to_text(
        "<title>Quarterly</title><style>p{color:red}</style>"
        "<p>Revenue fell.</p><script>tracker()</script><p>Costs rose.</p>"
    )
    assert title == "Quarterly"
    assert "Revenue fell." in text and "Costs rose." in text
    assert "tracker" not in text and "color:red" not in text


def test_a_link_must_be_http():
    with pytest.raises(A.AttachmentError, match="http"):
        A.build("link", url="ftp://example.com/file")


def test_a_photo_must_be_an_image_type():
    with pytest.raises(A.AttachmentError, match="PNG"):
        A.build("image", name="sheet.bmp", data_b64=b64(b"data"))


def test_a_photo_keeps_its_bytes_and_media_type():
    item = A.build("image", name="chart.png", data_b64=b64(b"pixels"))
    assert item.kind == "image" and item.media_type == "image/png"
    assert base64.b64decode(item.data_b64) == b"pixels"


def test_a_pdf_that_cannot_be_read_explains_itself():
    """Whether pypdf is present or not, the listener gets a sentence.

    The import is guarded with BaseException on purpose: a broken native crypto
    dependency raises pyo3's PanicException, which is not an Exception.
    """
    with pytest.raises(A.AttachmentError) as caught:
        A.extract_document("scan.pdf", b"%PDF-1.4 not really a pdf")
    assert "PDF" in str(caught.value)


# --- storage --------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return A.AttachmentStore(str(tmp_path / "attach.db"))


def test_an_attachment_belongs_to_one_listener(store):
    item = store.put("me", A.build("image", name="a.png", data_b64=b64(b"x")))
    assert store.get("me", item.id) is not None
    assert store.get("someone-else", item.id) is None, "attachments must not be shared"


def test_resolve_keeps_the_order_asked_for(store):
    first = store.put("me", A.build("image", name="a.png", data_b64=b64(b"1")))
    second = store.put("me", A.build("image", name="b.png", data_b64=b64(b"2")))
    got = store.resolve("me", [second.id, first.id])
    assert [g.id for g in got] == [second.id, first.id]


def _age(item: A.Attachment, seconds: float) -> A.Attachment:
    """Backdate an attachment. Ageing the row beats patching the clock."""
    item.created = time.time() - seconds
    return item


def test_an_expired_attachment_is_not_returned(store):
    item = A.build("image", name="a.png", data_b64=b64(b"x"))
    store.put("me", _age(item, A.TTL_SECONDS + 60))
    assert store.get("me", item.id) is None


def test_purge_removes_only_the_expired(store):
    fresh = store.put("me", A.build("image", name="a.png", data_b64=b64(b"x")))
    stale = A.build("image", name="b.png", data_b64=b64(b"y"))
    store.put("me", _age(stale, A.TTL_SECONDS + 60))
    assert store.purge_expired() == 1
    assert store.get("me", fresh.id) is not None


# --- the prompt -----------------------------------------------------------

def test_attached_text_reaches_the_prompt():
    item = A.Attachment(id="1", kind="document", name="q3.txt",
                        text="Revenue fell 12 percent.")
    prompt = build_prompt(plan_episode("what does this say", 3, attachments=(item,)))
    assert "Revenue fell 12 percent." in prompt
    assert 'name="q3.txt"' in prompt


def test_the_prompt_says_the_listener_material_outranks_recall():
    """Otherwise the model writes the version it already knows."""
    item = A.Attachment(id="1", kind="document", name="q3.txt", text="Margin was 41 percent.")
    prompt = build_prompt(plan_episode("summarise this", 3, attachments=(item,)))
    assert "outranks" in prompt


def test_photos_are_announced_but_not_pasted_as_text():
    photo = A.build("image", name="chart.png", data_b64=b64(b"pixels"))
    prompt = build_prompt(plan_episode("what is in this", 3, attachments=(photo,)))
    assert "attached 1 image" in prompt
    assert "pixels" not in prompt, "image bytes must travel as an image block"


def test_a_photo_becomes_an_image_block_before_the_instructions():
    """The prompt refers to the images, so they must already be in view."""
    from script_generator import ScriptGenerator

    photo = A.build("image", name="chart.png", data_b64=b64(b"pixels"))
    plan = plan_episode("what is in this", 3, attachments=(photo,))
    content = ScriptGenerator(api_key="")._request_kwargs(plan)["messages"][0]["content"]
    assert content[0]["type"] == "image"
    assert content[0]["source"]["media_type"] == "image/png"
    assert content[-1]["type"] == "text"


def test_no_attachments_leaves_the_prompt_untouched():
    plain = build_prompt(plan_episode("why is the sky blue", 3))
    assert "attached" not in plain.lower()
