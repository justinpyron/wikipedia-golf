# wikipedia-golf

AI agent that plays [Wikipedia Golf](https://en.wikipedia.org/wiki/Wikipedia:Wiki_Game): the game where you navigate from one article to another using the fewest links.

# How it works

- **Frontend** (`app.py`) — article search, model picker, run controls
- **Agent** (`agent.py`, `prompts.py`) — pydantic-ai agent with a `get_links` tool; moves are validated against game rules
- **Wikipedia** (`wiki.py`) — article content and outgoing links from the Wikimedia Core REST API
- **Observability** — [Logfire](https://logfire.pydantic.dev/) instrumentation

Inference runs in-process with the Dash app (no separate backend service).

# Project Organization

```
├── README.md
├── app.py                                  # Dash web app
├── agent.py                                # Agent factory, tool, cost estimation
├── wiki.py                                 # Wikipedia REST client
├── prompts.py                              # Versioned system prompts
├── assets/                                 # CSS, logos, favicons
├── evals/                                  # Benchmarks and leaderboard runs
├── .github/workflows/build-and-deploy.yml  # Cloud Run deploy
├── Dockerfile
├── pyproject.toml / uv.lock
```

# Installation

This project uses [uv](https://github.com/astral-sh/uv) (Python 3.12+).

1. Install `uv`

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. Install dependencies

```bash
uv sync
```

# Configuration

Set API keys for the providers you use:

| Env Variable | Provider |
|----------|----------|
| `OPENAI_API_KEY` | OpenAI |
| `ANTHROPIC_API_KEY` | Anthropic |
| `GOOGLE_API_KEY` | Google (Gemini) |
| `TOGETHER_API_KEY` | Together (Kimi, GLM, …) |
| `XAI_API_KEY` | xAI (Grok) |
| `LOGFIRE_TOKEN` | Logfire (optional) |
| `LOGFIRE_ENV` | `dev` or `prod` (default: `dev`) |

# Usage

```bash
uv run python app.py
```

Open [http://localhost:8080](http://localhost:8080), set origin and destination, pick a model, and run.
