#!/usr/bin/env python3
"""
Offline proof-of-concept for neoimage's image-to-image editing + closed visual loop.

Runs with NO network and NO API key: a stubbed Gemini endpoint (httpx.MockTransport)
stands in for the real API. It proves three things end-to-end:

  1. The edit request actually carries the input image(s) back to the model
     (inlineData parts) -- i.e. this is genuine image-to-image, not text-to-image.
  2. The MCP tool returns the edited image as a VIEWABLE image content block
     (mime image/png), so the calling model can see what it made and iterate.
  3. The edited bytes are also persisted to disk, and structured metadata
     (reference_count, model, file_path, text_response) comes back alongside.

Exit code 0 == all assertions passed.

    ./venv/bin/python demo_edit.py
"""

import asyncio
import base64
import json
import os
import sys
import tempfile

import httpx

import image_edit

# A 1x1 PNG used both as the "input" image and as the stubbed model "output".
PNG_1X1 = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
    "de0000000c4944415408d763f8cfc0f01f0005010102a0bb3d630000000049454e44ae426082"
)

# Distinct bytes for the "edited" result so we can prove we got the model's
# output back, not just echoed our input.
EDITED_PNG = bytes.fromhex(
    "89504e470d0a1a0a0000000d4948445200000001000000010802000000907753"
    "de0000000c4944415408d76368606000000005000157a4d2b30000000049454e44ae426082"
)

captured_request: dict = {}


def gemini_mock_handler(request: httpx.Request) -> httpx.Response:
    """Stand-in for generativelanguage.googleapis.com generateContent."""
    body = json.loads(request.content)
    captured_request["body"] = body
    parts = body["contents"][0]["parts"]
    ref_count = sum(1 for p in parts if "inlineData" in p)
    # Refuse to "edit" if no image was sent -- this stub only honors image-to-image.
    if ref_count == 0:
        return httpx.Response(400, json={"error": "no input image"})
    response = {
        "candidates": [
            {
                "content": {
                    "parts": [
                        {"text": f"Applied edit using {ref_count} reference image(s)."},
                        {
                            "inlineData": {
                                "mimeType": "image/png",
                                "data": base64.b64encode(EDITED_PNG).decode("ascii"),
                            }
                        },
                    ]
                }
            }
        ]
    }
    return httpx.Response(200, json=response)


def check(label: str, condition: bool) -> None:
    status = "ok " if condition else "FAIL"
    print(f"  [{status}] {label}")
    if not condition:
        raise SystemExit(f"assertion failed: {label}")


async def main() -> None:
    # --- Part A: pure-logic guards (no network at all) ------------------------
    print("Part A -- request construction & validation (pure):")
    ref = image_edit.ReferenceImage(mime_type="image/png", data=PNG_1X1)
    body = image_edit.build_gemini_edit_body("make it pop", [ref])
    parts = body["contents"][0]["parts"]
    check("prompt is the first part", parts[0] == {"text": "make it pop"})
    check("input image attached as inlineData", "inlineData" in parts[1])
    check("mime detected from bytes, not extension", image_edit.detect_mime(PNG_1X1) == "image/png")

    raised = False
    try:
        image_edit.build_gemini_edit_body("x", [])
    except ValueError:
        raised = True
    check("editing with zero references is rejected", raised)

    raised = False
    try:
        image_edit.build_gemini_edit_body("x", [ref] * (image_edit.MAX_REFERENCE_IMAGES + 1))
    except ValueError:
        raised = True
    check(f"more than {image_edit.MAX_REFERENCE_IMAGES} references is rejected", raised)

    # --- Part B: full edit round-trip via mocked transport --------------------
    print("Part B -- image-to-image round-trip (mocked Gemini, no network):")
    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "input.png")
        with open(in_path, "wb") as f:
            f.write(PNG_1X1)

        mock_client = httpx.AsyncClient(transport=httpx.MockTransport(gemini_mock_handler))
        result = await image_edit.gemini_edit(
            prompt="swap the background to a beach",
            reference_paths=[in_path],
            api_key="offline-test-key",
            http_client=mock_client,
        )
        check("model received the input image", captured_request["body"]
              ["contents"][0]["parts"][1].get("inlineData") is not None)
        check("edited bytes are the model output (not the input)", result.image_bytes == EDITED_PNG)
        check("reference_count reported", result.reference_count == 1)
        check("model text response captured", "reference image" in (result.text_response or ""))

    # --- Part C: closed visual loop through the real MCP tool -----------------
    print("Part C -- MCP tool returns a VIEWABLE image (closed loop):")
    # Inject the mock transport into the tool's HTTP factory so the real
    # edit_image_gemini tool runs fully offline.
    image_edit._make_client = lambda timeout: httpx.AsyncClient(
        transport=httpx.MockTransport(gemini_mock_handler), timeout=timeout
    )
    os.environ["GOOGLE_API_KEY"] = "offline-test-key"

    from fastmcp import Client
    import server

    with tempfile.TemporaryDirectory() as tmp:
        in_path = os.path.join(tmp, "portrait.png")
        out_path = os.path.join(tmp, "edited.png")
        with open(in_path, "wb") as f:
            f.write(PNG_1X1)

        async with Client(server.mcp) as client:
            tools = {t.name for t in await client.list_tools()}
            check("edit_image_gemini is registered as an MCP tool", "edit_image_gemini" in tools)

            res = await client.call_tool(
                "edit_image_gemini",
                {
                    "prompt": "give the character a red jacket, keep the face identical",
                    "input_images": [in_path],
                    "output_path": out_path,
                },
            )

        image_blocks = [b for b in res.content if getattr(b, "type", None) == "image"]
        check("tool returned a viewable image content block", len(image_blocks) == 1)
        check("returned image is a PNG", image_blocks[0].mimeType == "image/png")
        decoded = base64.b64decode(image_blocks[0].data)
        check("viewable image is the edited output", decoded == EDITED_PNG)
        check("edited image also written to disk", os.path.exists(out_path))
        with open(out_path, "rb") as f:
            check("file on disk matches edited bytes", f.read() == EDITED_PNG)
        check("structured metadata reports reference_count",
              res.structured_content.get("reference_count") == 1)

    print("\nAll checks passed -- image-to-image editing + closed visual loop proven offline.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except SystemExit as exc:
        print(exc, file=sys.stderr)
        sys.exit(1)
    sys.exit(0)
