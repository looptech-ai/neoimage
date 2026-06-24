"""Offline unit/integration tests for the Gemini image-edit path.

No network and no API key required -- the HTTP layer is stubbed with
httpx.MockTransport. Run with:  ./venv/bin/python -m pytest tests/ -q
"""

import base64
import json
import os
import sys

import httpx
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import image_edit  # noqa: E402

PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
    "de0000000c4944415408d763f8cfc0f01f0005010102a0bb3d630000000049454e44ae426082"
)
JPEG_HEADER = b"\xff\xd8\xff\xe0\x00\x10JFIF"
WEBP_HEADER = b"RIFF\x00\x00\x00\x00WEBPVP8 "
EDITED_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
    "de0000000c4944415408d76368606000000005000157a4d2b30000000049454e44ae426082"
)


# --- pure helpers ----------------------------------------------------------

@pytest.mark.parametrize(
    "data,expected",
    [
        (PNG_1X1, "image/png"),
        (JPEG_HEADER, "image/jpeg"),
        (WEBP_HEADER, "image/webp"),
        (b"GIF89a....", "image/gif"),
    ],
)
def test_detect_mime_from_magic_bytes(data, expected):
    assert image_edit.detect_mime(data) == expected


def test_detect_mime_rejects_unknown():
    with pytest.raises(ValueError):
        image_edit.detect_mime(b"not an image")
    with pytest.raises(ValueError):
        image_edit.detect_mime(b"")


def test_build_body_interleaves_prompt_and_images():
    ref = image_edit.ReferenceImage(mime_type="image/png", data=PNG_1X1)
    body = image_edit.build_gemini_edit_body("edit me", [ref, ref])
    parts = body["contents"][0]["parts"]
    assert parts[0] == {"text": "edit me"}
    assert sum(1 for p in parts if "inlineData" in p) == 2
    # image bytes are base64-encoded into the request
    assert parts[1]["inlineData"]["data"] == base64.b64encode(PNG_1X1).decode()
    assert parts[1]["inlineData"]["mimeType"] == "image/png"
    assert body["generationConfig"]["responseModalities"] == ["IMAGE", "TEXT"]


def test_build_body_requires_at_least_one_reference():
    with pytest.raises(ValueError, match="at least one reference"):
        image_edit.build_gemini_edit_body("x", [])


def test_build_body_caps_reference_count():
    ref = image_edit.ReferenceImage(mime_type="image/png", data=PNG_1X1)
    too_many = [ref] * (image_edit.MAX_REFERENCE_IMAGES + 1)
    with pytest.raises(ValueError, match="too many reference images"):
        image_edit.build_gemini_edit_body("x", too_many)


def test_build_body_rejects_empty_prompt():
    ref = image_edit.ReferenceImage(mime_type="image/png", data=PNG_1X1)
    with pytest.raises(ValueError, match="non-empty"):
        image_edit.build_gemini_edit_body("   ", [ref])


def test_build_body_thinking_only_for_nb2():
    ref = image_edit.ReferenceImage(mime_type="image/png", data=PNG_1X1)
    nb2 = image_edit.build_gemini_edit_body(
        "x", [ref], model="gemini-3.1-flash-image-preview", thinking_level="high"
    )
    assert nb2["generationConfig"]["thinkingConfig"]["thinkingLevel"] == "high"
    pro = image_edit.build_gemini_edit_body(
        "x", [ref], model="gemini-3-pro-image-preview", thinking_level="high"
    )
    assert "thinkingConfig" not in pro["generationConfig"]


def test_parse_response_extracts_image_and_text():
    payload = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": "done"},
                        {"inlineData": {"data": base64.b64encode(EDITED_PNG).decode()}},
                    ]
                }
            }
        ]
    }
    img, text = image_edit.parse_gemini_image_response(payload)
    assert img == EDITED_PNG
    assert text == "done"


def test_parse_response_raises_without_image():
    with pytest.raises(ValueError):
        image_edit.parse_gemini_image_response({"candidates": [{"content": {"parts": []}}]})


# --- full round-trip via mocked transport ----------------------------------

def _mock_client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


@pytest.mark.asyncio
async def test_gemini_edit_round_trip(tmp_path):
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        seen["api_key"] = request.headers.get("x-goog-api-key")
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "parts": [
                                {"text": "edited"},
                                {"inlineData": {"data": base64.b64encode(EDITED_PNG).decode()}},
                            ]
                        }
                    }
                ]
            },
        )

    in_path = tmp_path / "in.png"
    in_path.write_bytes(PNG_1X1)

    result = await image_edit.gemini_edit(
        prompt="recolor",
        reference_paths=[str(in_path)],
        api_key="k-123",
        http_client=_mock_client(handler),
    )

    # the input image really went to the model
    assert "inlineData" in seen["body"]["contents"][0]["parts"][1]
    assert seen["api_key"] == "k-123"
    assert result.image_bytes == EDITED_PNG
    assert result.reference_count == 1
    assert result.text_response == "edited"


@pytest.mark.asyncio
async def test_gemini_edit_surfaces_api_error():
    def handler(request):
        return httpx.Response(403, text="permission denied")

    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(PNG_1X1)
        path = f.name
    try:
        with pytest.raises(ValueError, match="403"):
            await image_edit.gemini_edit(
                prompt="x",
                reference_paths=[path],
                api_key="k",
                http_client=_mock_client(handler),
            )
    finally:
        os.unlink(path)
