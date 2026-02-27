# neoimage

An MCP server for AI image generation, providing Claude Code and other MCP-compatible clients access to **Google Gemini** (Imagen) and **OpenAI** (DALL-E / GPT Image) models via simple tool calls.

## Tools

### `generate_image_gemini`
Generate images using Google's Gemini / Imagen models. Defaults to **Nano Banana 2** (`gemini-3.1-flash-image-preview`), Google's latest image model with 4K support, character consistency, and thinking mode.

| Parameter | Options | Default | Notes |
|---|---|---|---|
| `prompt` | string | required | Describe the image you want. NB2 supports up to 5 consistent characters and 14 objects. |
| `model` | `gemini-3.1-flash-image-preview` · `gemini-2.5-flash-image` · `gemini-3-pro-image-preview` | `gemini-3.1-flash-image-preview` | NB2 (default) is fastest + highest quality. Use pro for production assets. |
| `aspect_ratio` | `1:1` · `16:9` · `9:16` · `4:3` · `3:4` · `3:2` · `2:3` · `5:4` · `4:5` · `21:9` · `4:1` · `1:4` · `8:1` · `1:8` | `1:1` | Ultra-wide/tall ratios (`4:1`, `8:1`, `1:4`, `1:8`) require NB2 |
| `image_size` | `512px` · `1K` · `2K` · `4K` | model default (1K) | `4K` adds latency (~15-25s); `512px` is fastest (~3-8s) |
| `thinking_level` | `minimal` · `high` | none | NB2 only. Improves composition at the cost of latency. |
| `output_path` | string | auto-generated | Full path e.g. `/tmp/my-image.png` |

### `generate_image_openai`
Generate images using OpenAI's GPT Image models.

| Parameter | Options | Default | Notes |
|---|---|---|---|
| `prompt` | string | required | Describe the image you want |
| `model` | `gpt-image-1` · `gpt-image-1-mini` · `gpt-image-1.5` | `gpt-image-1` | Use 1.5 for highest quality |
| `size` | `1024x1024` · `1536x1024` · `1024x1536` · `auto` | `1024x1024` | |
| `format` | `png` · `jpeg` · `webp` | `png` | |
| `background` | `opaque` · `transparent` · `auto` | `opaque` | `transparent` for logos (PNG only) |
| `quality` | `auto` · `low` · `medium` · `high` | `auto` | |
| `output_path` | string | auto-generated | Full path e.g. `/tmp/my-image.png` |

---

## Setup

### 1. Clone the repo

```bash
git clone https://github.com/looptech-ai/neoimage.git
cd neoimage
```

### 2. Create a virtual environment and install dependencies

```bash
python3 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Get API keys

You need at least one of:

- **Google API key** (for Gemini) — [aistudio.google.com/apikey](https://aistudio.google.com/apikey)
- **OpenAI API key** (for GPT Image) — [platform.openai.com/api-keys](https://platform.openai.com/api-keys)

### 4. Configure Claude Code

Copy `.mcp.json.example` to `.mcp.json` in the **project directory where you want to use neoimage**, then fill in your paths and keys:

```bash
cp .mcp.json.example /path/to/your/project/.mcp.json
```

Edit the file:

```json
{
  "mcpServers": {
    "neoimage": {
      "command": "/absolute/path/to/neoimage/venv/bin/fastmcp",
      "args": ["run", "/absolute/path/to/neoimage/server.py"],
      "env": {
        "GOOGLE_API_KEY": "your-google-api-key",
        "OPENAI_API_KEY": "your-openai-api-key"
      }
    }
  }
}
```

> **Paths must be absolute.** Replace `/absolute/path/to/neoimage` with the actual path where you cloned this repo (e.g. `/Users/yourname/dev/neoimage`).

### 5. Verify it's working

In Claude Code, you should see `neoimage` listed as an available MCP server. Try:

```
Generate a 16:9 dark slide background at 2K resolution
```

---

## Example usage in Claude Code

```
Generate a professional dark presentation slide on a #262626 background
titled "Q1 Results" with a bar chart showing revenue growth.
Use gemini-3-pro-image-preview, aspect ratio 16:9,
save to /tmp/slide-q1.png
```

---

## Tips

- **Best all-round**: `gemini-3.1-flash-image-preview` (Nano Banana 2, default) — fast + high quality
- **Presentation slides**: `aspect_ratio: 16:9` with `image_size: 2K` or `4K`
- **Complex compositions**: Add `thinking_level: high` for better instruction following
- **Quick drafts**: `image_size: 512px` for fastest generation (~3-8s)
- **Ultra-wide banners**: `aspect_ratio: 8:1` or `4:1` (NB2 only)
- **Logos / icons with transparency**: `generate_image_openai` with `background: transparent`, `format: png`
- **High-detail illustrations**: `gpt-image-1.5` with `quality: high`
- Generated images save to the **current working directory** unless you specify `output_path`

---

## Security

- Never commit `.mcp.json` — it's in `.gitignore` and contains your API keys
- Use `.mcp.json.example` as a template (safe to commit — contains no real keys)
- Each team member uses their own API keys

---

## Stack

- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP server framework
- [Google Generative Language API](https://ai.google.dev/) — Gemini / Imagen
- [OpenAI Python SDK](https://github.com/openai/openai-python) — GPT Image
