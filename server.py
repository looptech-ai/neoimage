"""
Image Generation MCP Server

Provides tools for generating images using Google Gemini and OpenAI APIs.

Gemini tool uses Nano Banana 2 (gemini-3.1-flash-image-preview) by default —
Google's latest image generation model with 4K resolution, expanded aspect ratios,
character consistency for up to 5 characters, and optional thinking mode.
"""

import base64
import os
import uuid
from datetime import datetime
from typing import Literal

import httpx
from fastmcp import FastMCP

# Initialize MCP server
mcp = FastMCP("neoimage")


def generate_output_path(prefix: str, extension: str = "png") -> str:
    """Generate a unique output path for an image."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"{prefix}_{timestamp}_{unique_id}.{extension}"


@mcp.tool()
async def generate_image_gemini(
    prompt: str,
    model: Literal[
        "gemini-3.1-flash-image-preview",
        "gemini-2.5-flash-image",
        "gemini-3-pro-image-preview",
    ] = "gemini-3.1-flash-image-preview",
    aspect_ratio: Literal[
        "1:1",
        "2:3", "3:2",
        "3:4", "4:3",
        "4:5", "5:4",
        "9:16", "16:9",
        "21:9",
        "4:1", "1:4",
        "8:1", "1:8",
    ] = "1:1",
    image_size: Literal["512px", "1K", "2K", "4K"] | None = None,
    thinking_level: Literal["minimal", "high"] | None = None,
    output_path: str | None = None,
) -> dict:
    """
    Generate an image using Google Gemini API (Nano Banana 2 by default).

    Default model is gemini-3.1-flash-image-preview (Nano Banana 2), released Feb 2026.
    It combines the quality of Nano Banana Pro with the speed of Gemini Flash.

    OUTPUT LOCATION: Images save to current working directory by default with auto-generated
    names like "gemini_20240126_143052_a1b2c3d4.png". Use output_path for custom location.

    MODELS:
    - gemini-3.1-flash-image-preview (default): Nano Banana 2 — fastest + highest quality,
      4K resolution, character consistency for up to 5 characters, 14-object fidelity,
      expanded aspect ratios, advanced text rendering, optional thinking mode
    - gemini-2.5-flash-image: Previous generation (Nano Banana 1), max 2K
    - gemini-3-pro-image-preview: Pro-tier quality for production assets

    ASPECT RATIOS (Nano Banana 2 adds ultra-wide/tall):
    - Square: 1:1
    - Landscape: 3:2, 4:3, 5:4, 16:9, 21:9
    - Portrait: 2:3, 3:4, 4:5, 9:16
    - Ultra-wide (NB2 only): 4:1, 8:1
    - Ultra-tall (NB2 only): 1:4, 1:8

    IMAGE SIZE (Nano Banana 2):
    - 512px: Minimum, fastest (~3-8s)
    - 1K: Default if not specified (~5-10s)
    - 2K: Balanced quality (~10-15s)
    - 4K: Maximum resolution (~15-25s)

    THINKING MODE (Nano Banana 2 only):
    - minimal (default when enabled): Light reasoning pass, small quality boost
    - high: Deep reasoning, best for complex scenes or precise compositions

    Args:
        prompt: Text description of the image to generate. Be descriptive for best results.
            For Nano Banana 2, you can describe up to 5 characters with consistent appearance
            and up to 14 distinct objects.
        model: Gemini model to use. Defaults to gemini-3.1-flash-image-preview (Nano Banana 2).
        aspect_ratio: Output dimensions ratio. 4:1/8:1/1:4/1:8 require Nano Banana 2.
        image_size: Output resolution (512px/1K/2K/4K). None uses the model default (1K).
            4K adds significant latency; use for final/hero images only.
        thinking_level: Enable thinking mode for Nano Banana 2 (minimal or high).
            Improves composition and instruction-following at the cost of latency.
            Only supported by gemini-3.1-flash-image-preview; ignored for other models.
        output_path: Full path to save image (e.g., "/path/to/image.png"). Auto-generates if not provided.

    Returns:
        Dictionary with file_path (absolute path to saved image), model, aspect_ratio,
        image_size, thinking_level, prompt, and any text_response from the model.
    """
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")

    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": api_key,
    }

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "responseModalities": ["IMAGE", "TEXT"],
        },
    }

    # Build imageConfig with aspect ratio and image size
    image_config = {}
    if aspect_ratio != "1:1":
        image_config["aspectRatio"] = aspect_ratio
    if image_size:
        image_config["imageSize"] = image_size
    if image_config:
        body["generationConfig"]["imageConfig"] = image_config

    # Thinking mode — Nano Banana 2 (gemini-3.1-flash-image-preview) only
    if thinking_level and model == "gemini-3.1-flash-image-preview":
        body["generationConfig"]["thinkingConfig"] = {
            "thinkingLevel": thinking_level,
            "includeThoughts": False,
        }

    # 4K + thinking mode can take up to 25s; use a generous timeout
    timeout = 300.0 if image_size == "4K" else 120.0

    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, headers=headers, json=body)

        if response.status_code != 200:
            error_detail = response.text
            raise ValueError(f"Gemini API error ({response.status_code}): {error_detail}")

        result = response.json()

    # Extract image data from response
    candidates = result.get("candidates", [])
    if not candidates:
        raise ValueError("No candidates returned from Gemini API")

    parts = candidates[0].get("content", {}).get("parts", [])

    image_data = None
    text_response = None

    for part in parts:
        if "inlineData" in part:
            image_data = part["inlineData"]["data"]
        elif "text" in part:
            text_response = part["text"]

    if not image_data:
        raise ValueError(f"No image data in response. Text response: {text_response}")

    # Decode and save image
    image_bytes = base64.b64decode(image_data)

    if output_path is None:
        output_path = generate_output_path("gemini", "png")

    with open(output_path, "wb") as f:
        f.write(image_bytes)

    return {
        "file_path": os.path.abspath(output_path),
        "model": model,
        "aspect_ratio": aspect_ratio,
        "image_size": image_size,
        "thinking_level": thinking_level,
        "prompt": prompt,
        "text_response": text_response,
    }


@mcp.tool()
async def generate_image_openai(
    prompt: str,
    model: Literal["gpt-image-1", "gpt-image-1-mini", "gpt-image-1.5"] = "gpt-image-1",
    size: Literal["1024x1024", "1536x1024", "1024x1536", "auto"] = "1024x1024",
    format: Literal["png", "jpeg", "webp"] = "png",
    background: Literal["opaque", "transparent", "auto"] = "opaque",
    quality: Literal["auto", "low", "medium", "high"] = "auto",
    output_path: str | None = None,
) -> dict:
    """
    Generate an image using OpenAI's DALL-E / GPT Image API. Powered by OpenAI's image models.

    OUTPUT LOCATION: Images save to current working directory by default with auto-generated
    names like "openai_20240126_143052_a1b2c3d4.png". Use output_path for custom location.

    MODELS (OpenAI):
    - gpt-image-1 (default): Standard quality, good balance of speed and quality
    - gpt-image-1-mini: Faster, lower cost, slightly reduced quality
    - gpt-image-1.5: Highest quality, best for detailed/complex images

    SIZES: 1024x1024 (square), 1536x1024 (landscape), 1024x1536 (portrait), auto (model decides)

    QUALITY: low (fastest), medium, high (best detail), auto (model decides based on prompt)

    BACKGROUND: opaque (solid background), transparent (PNG only, for logos/icons), auto

    Args:
        prompt: Text description of the image. OpenAI may revise your prompt for better results.
        model: Which OpenAI model - use gpt-image-1.5 for quality, gpt-image-1-mini for speed
        size: Pixel dimensions - 1536x1024 for landscape, 1024x1536 for portrait
        format: Output format - png (lossless), jpeg (smaller), webp (modern/efficient)
        background: Use transparent for logos/icons that need no background (PNG only)
        quality: Higher quality = more detail but slower. Use high for important images.
        output_path: Full path to save image. Auto-generates if not provided.

    Returns:
        Dictionary with file_path (absolute path), model, size, format, quality, prompt, and revised_prompt if modified
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set")

    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=api_key)

    # Build generation parameters
    params = {
        "model": model,
        "prompt": prompt,
        "n": 1,
    }

    # Add optional parameters
    if size != "auto":
        params["size"] = size
    if quality != "auto":
        params["quality"] = quality
    if background != "opaque":
        params["background"] = background

    response = await client.images.generate(**params)

    if not response.data:
        raise ValueError("No image data returned from OpenAI API")

    # Get image URL and download it
    image_url = response.data[0].url
    if not image_url:
        raise ValueError("No image URL in response")

    async with httpx.AsyncClient(timeout=60.0) as http_client:
        img_response = await http_client.get(image_url)
        if img_response.status_code != 200:
            raise ValueError(f"Failed to download image: {img_response.status_code}")
        image_bytes = img_response.content

    if output_path is None:
        output_path = generate_output_path("openai", format)

    with open(output_path, "wb") as f:
        f.write(image_bytes)

    return {
        "file_path": os.path.abspath(output_path),
        "model": model,
        "size": size,
        "format": format,
        "background": background,
        "quality": quality,
        "prompt": prompt,
        "revised_prompt": getattr(response.data[0], "revised_prompt", None),
    }


if __name__ == "__main__":
    mcp.run()
