"""
Image-to-image editing for Google Gemini (Nano Banana 2).

This is the capability the generation-only `server.py` never built: feeding one or
more *reference images* back into the model so it can edit, compose, or hold a
character consistent across renders -- the headline feature of the Gemini image
models that the README advertises ("character consistency for up to 5 characters")
but the existing tools cannot actually exercise, because they only ever send text.

The functions here are deliberately split into:
  * pure, network-free helpers (mime detection, request-body construction,
    response parsing) that can be unit-tested offline with no API key, and
  * a single async `gemini_edit()` with an injectable HTTP client so the whole
    request/response round-trip is exercisable against `httpx.MockTransport`.

Nothing here imports FastMCP -- the MCP tool wrapper lives in `server.py`.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass

import httpx

# Gemini Nano Banana 2 accepts up to 14 reference images in a single request
# (and can hold up to 5 of them consistent as "characters").
MAX_REFERENCE_IMAGES = 14

DEFAULT_MODEL = "gemini-3.1-flash-image-preview"
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
)

# Magic-byte signatures -> MIME type. Order matters only in that each check is
# unambiguous; we never guess.
_MAGIC = (
    (b"\x89PNG\r\n\x1a\n", "image/png"),
    (b"\xff\xd8\xff", "image/jpeg"),
    (b"GIF87a", "image/gif"),
    (b"GIF89a", "image/gif"),
)


@dataclass(frozen=True)
class ReferenceImage:
    """A decoded input image ready to be attached to a request."""

    mime_type: str
    data: bytes

    @property
    def as_inline_part(self) -> dict:
        return {
            "inlineData": {
                "mimeType": self.mime_type,
                "data": base64.b64encode(self.data).decode("ascii"),
            }
        }


@dataclass(frozen=True)
class EditResult:
    """The product of an edit: raw image bytes plus provenance metadata."""

    image_bytes: bytes
    mime_type: str
    text_response: str | None
    model: str
    reference_count: int


def detect_mime(data: bytes) -> str:
    """Detect an image MIME type from its magic bytes. Trusts no file extension.

    Raises ValueError for unrecognized/empty input rather than guessing -- the
    request must declare a real mimeType or the model will reject it.
    """
    if not data:
        raise ValueError("empty image data")
    for signature, mime in _MAGIC:
        if data.startswith(signature):
            return mime
    # WebP is RIFF<size>WEBP, so the marker sits at byte 8, not byte 0.
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    raise ValueError("unrecognized image format (not PNG/JPEG/WebP/GIF)")


def load_reference_image(path: str) -> ReferenceImage:
    """Read an image file from disk and sniff its MIME type from content."""
    with open(path, "rb") as f:
        data = f.read()
    return ReferenceImage(mime_type=detect_mime(data), data=data)


def build_gemini_edit_body(
    prompt: str,
    references: list[ReferenceImage],
    *,
    model: str = DEFAULT_MODEL,
    aspect_ratio: str = "1:1",
    image_size: str | None = None,
    thinking_level: str | None = None,
) -> dict:
    """Construct the `generateContent` request body for an image edit.

    The shape mirrors the text-only body in `server.py` but interleaves the
    reference images as `inlineData` parts alongside the prompt text -- which is
    exactly what turns "generate" into "edit/compose/keep-consistent".
    """
    if not prompt or not prompt.strip():
        raise ValueError("prompt must be a non-empty string")
    if not references:
        raise ValueError(
            "edit requires at least one reference image; "
            "use generate_image_gemini for text-only generation"
        )
    if len(references) > MAX_REFERENCE_IMAGES:
        raise ValueError(
            f"too many reference images: {len(references)} "
            f"(Gemini accepts at most {MAX_REFERENCE_IMAGES})"
        )

    parts: list[dict] = [{"text": prompt}]
    parts.extend(ref.as_inline_part for ref in references)

    body: dict = {
        "contents": [{"parts": parts}],
        "generationConfig": {"responseModalities": ["IMAGE", "TEXT"]},
    }

    image_config: dict = {}
    if aspect_ratio != "1:1":
        image_config["aspectRatio"] = aspect_ratio
    if image_size:
        image_config["imageSize"] = image_size
    if image_config:
        body["generationConfig"]["imageConfig"] = image_config

    # thinking mode is Nano Banana 2 only; silently irrelevant elsewhere.
    if thinking_level and model == "gemini-3.1-flash-image-preview":
        body["generationConfig"]["thinkingConfig"] = {
            "thinkingLevel": thinking_level,
            "includeThoughts": False,
        }

    return body


def parse_gemini_image_response(result: dict) -> tuple[bytes, str | None]:
    """Pull the image bytes (and any text) out of a generateContent response.

    Same envelope the existing generator parses, factored out so the edit path
    and a future refactor of the generate path can share one parser.
    """
    candidates = result.get("candidates", [])
    if not candidates:
        raise ValueError("no candidates returned from Gemini API")

    parts = candidates[0].get("content", {}).get("parts", [])
    image_data: str | None = None
    text_response: str | None = None
    for part in parts:
        if "inlineData" in part:
            image_data = part["inlineData"]["data"]
        elif "text" in part:
            text_response = part["text"]

    if not image_data:
        raise ValueError(
            f"no image data in response. Text response: {text_response}"
        )
    return base64.b64decode(image_data), text_response


def _make_client(timeout: float) -> httpx.AsyncClient:
    """HTTP client factory. Overridable in tests/demos to inject a MockTransport
    so the full edit round-trip runs with zero network and no API key."""
    return httpx.AsyncClient(timeout=timeout)


async def gemini_edit(
    *,
    prompt: str,
    reference_paths: list[str],
    api_key: str,
    model: str = DEFAULT_MODEL,
    aspect_ratio: str = "1:1",
    image_size: str | None = None,
    thinking_level: str | None = None,
    http_client: httpx.AsyncClient | None = None,
) -> EditResult:
    """Edit/compose an image from a prompt plus one or more reference images.

    Pass `http_client` to inject a transport (offline testing); otherwise a real
    client is created and closed for you.
    """
    references = [load_reference_image(p) for p in reference_paths]
    body = build_gemini_edit_body(
        prompt,
        references,
        model=model,
        aspect_ratio=aspect_ratio,
        image_size=image_size,
        thinking_level=thinking_level,
    )

    url = GEMINI_ENDPOINT.format(model=model)
    headers = {"Content-Type": "application/json", "x-goog-api-key": api_key}
    timeout = 300.0 if image_size == "4K" else 120.0

    client = http_client or _make_client(timeout)
    owns_client = http_client is None
    try:
        response = await client.post(url, headers=headers, json=body)
        if response.status_code != 200:
            raise ValueError(
                f"Gemini API error ({response.status_code}): {response.text}"
            )
        result = response.json()
    finally:
        if owns_client:
            await client.aclose()

    image_bytes, text_response = parse_gemini_image_response(result)
    return EditResult(
        image_bytes=image_bytes,
        mime_type=detect_mime(image_bytes),
        text_response=text_response,
        model=model,
        reference_count=len(references),
    )
