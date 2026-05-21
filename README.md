# wikipedia-golf

AI agent that plays [Wikipedia Golf](https://en.wikipedia.org/wiki/Wikipedia:Wiki_Game)—navigate from one article to another using the fewest links. A **Dash** web app sets start/end articles and runs a **pydantic-ai** agent that fetches pages via the **Wikimedia REST API**.

**What is Wikipedia Golf?**

Pick an origin and destination article. Each hop must follow a link on the current page. Fewer hops wins. In this app, an LLM agent plays the game; you choose the articles and model, then watch the path, timing, and estimated cost.

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
