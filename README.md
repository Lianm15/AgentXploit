# AgentXploit

AgentXploit is a Python project with:
- a FastAPI backend (`backend/`)
- a Streamlit frontend (`frontend/`)

## Prerequisites

### Docker (recommended)

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (includes Docker Compose)
- A Gemini API key (Google AI Studio)

### Manual setup

- Python 3.10+ (3.11 recommended)
- `pip`
- Ollama installed and running locally
- A Gemini API key

## Project Structure

```text
AgentXploit/
├── backend/
│   ├── main.py              FastAPI app & CORS
│   ├── routes.py            REST API endpoints
│   ├── logic.py             Attack orchestration loop
│   ├── gemini.py            Gemini attacker + function-calling integration
│   ├── tools.py             Real server-side jailbreak tools
│   ├── pyrit_crescendo.py   PyRIT CrescendoOrchestrator subprocess helper
│   └── database.py          SQLite schema & connection
├── frontend/
│   ├── app.py         Streamlit UI (home + live transcript)
│   ├── api_client.py  HTTP client to backend
│   └── styles/        CSS for home, chat, stats pages
├── requirements.txt
├── Dockerfile         Multi-stage build (backend + frontend)
├── docker-compose.yml Orchestrates Ollama, backend, frontend
└── README.md
```

---

## Attack Tools

AgentXploit uses Gemini function calling to trigger real server-side tools during an attack session.
Gemini decides on its own when to call each tool based on what new information it needs — tools
are not called automatically at startup. Each tool result is sent back to Gemini as a
`FunctionResponse` and stays in the conversation history for the rest of the session.

The tools are defined in [`backend/tools.py`](backend/tools.py) and exposed to Gemini via the
Google Generative AI function-calling API in [`backend/gemini.py`](backend/gemini.py).

---

### Tool 1 — `get_model_profile` · Ollama Model Intelligence

**What it does**
Calls Ollama's `/api/show` API to retrieve the real technical details of the target model:
architecture family, parameter count, quantization level, and the exact prompt template format
used by that model variant. Also returns a curated set of documented attack vectors for that
model family, sourced from published red-team research (HarmBench, JailbreakBench benchmark
results, and community findings).

**Why it matters for jailbreaking**
Different model families have completely different safety training levels and known weak spots.
The prompt template delimiters (`[INST]<<SYS>>`, `###System:`, `<|system|>`, etc.) can be used
directly for template-injection attacks — but only if you know which format the model uses.

**When Gemini uses it**
Called when the model's family, template format, or known weak spots are not yet known. Skipped
if the model has already been profiled earlier in the session.

---

### Tool 2 — `run_garak_probe` · NVIDIA Garak LLM Scanner

**What it does**
Runs [NVIDIA's Garak](https://github.com/NVIDIA/garak) LLM vulnerability scanner as a subprocess
inside the backend container, pointing it at the same Ollama service that the attack loop uses.
Garak fires real attack payloads from its probe library against the target model and reports
exactly which attempts passed (model complied) and which failed (model refused).

Available probes:

| Probe name | What it tests |
|---|---|
| `dan` | DAN-style persona override attacks |
| `encoding_base64` | Base64-encoded payload injection |
| `encoding_rot13` | ROT13-encoded bypass |
| `encoding_morse` | Morse code encoding bypass |
| `encoding_braille` | Braille encoding bypass |
| `prompt_inject` | Prompt injection via context manipulation |
| `continuation` | Unsafe sentence continuation |
| `malwaregen` | Malware generation resistance |
| `xss` | XSS via markdown rendering |
| `leakreplay` | Training data memorization / replay |

**Why it matters for jailbreaking**
Instead of guessing which attack category to use and spending 10+ attempts finding out, Garak
gives you empirical pass/fail results in one call. If `encoding_base64` shows the model is
vulnerable, Gemini knows with certainty to use base64-encoded payloads next.

**When Gemini uses it**
Called when initial attacks are failing and the agent needs hard evidence of which technique
category actually works. Also called before committing to an encoding-based attack.

---

### Tool 3 — `apply_pyrit_converter` · Microsoft PyRIT Encoders

**What it does**
Transforms a prompt using one of Microsoft [PyRIT](https://github.com/Azure/PyRIT)'s built-in
converters to bypass keyword-based safety filters. Converters available:

| Converter | What it does |
|---|---|
| `base64` | Encodes the full prompt in Base64 |
| `rot13` | ROT13 letter substitution |
| `unicode_sub` | Substitutes characters with Unicode lookalikes |
| `leetspeak` | Numeric/symbol character substitution |
| `ascii_art` | Converts text to ASCII art |
| `char_space` | Inserts spaces between every character |
| `noise` | Adds random noise characters |

**Why it matters for jailbreaking**
Keyword-based safety filters operate on the raw text of a prompt. Encoding the sensitive portion
means pattern-matching safety systems cannot detect what is being asked — while the model can
still decode and respond to the content.

**When Gemini uses it**
Called when a good attack prompt is already formed but keeps being caught by keyword filters.
Most effective after Garak confirms the model is vulnerable to that encoding type.

---

### Tool 4 — `run_pyrit_crescendo` · Microsoft PyRIT Multi-Turn Escalation

**What it does**
Runs Microsoft [PyRIT](https://github.com/Azure/PyRIT)'s `CrescendoOrchestrator` — an automated
multi-turn gradual escalation attack. The orchestrator starts with a completely benign opening,
then incrementally escalates each turn toward the target behavior, exploiting the fact that prior
compliance creates implicit permission for the next step.

The orchestrator is launched via a subprocess helper (`pyrit_crescendo.py`) that connects to
Ollama's OpenAI-compatible `/v1` endpoint.

**Why it matters for jailbreaking**
Many models have strong single-turn refusal training but poor multi-turn consistency. A model
that would refuse the target question outright will often comply when it is reached gradually
after several innocuous turns have already been accepted.

**When Gemini uses it**
Called when direct single-turn attacks keep failing and the model is suspected to have weak
multi-turn consistency.

---

### Tool 5 — `fetch_jailbreak_prompts` · JailbreakBench Academic Dataset

**What it does**
Performs a live HTTP fetch of the [JailbreakBench](https://jailbreakbench.github.io/) public
dataset hosted on GitHub. JailbreakBench is the leading academic jailbreak benchmark with 100
harmful behaviors and verified attack prompts that successfully broke GPT-4, Claude 3, and
Gemini 1.5 in peer-reviewed evaluations (NeurIPS 2024).

The tool finds the closest matching behaviors to the current attack goal using keyword overlap
scoring, and when available, fetches the actual proven PAIR prompt that broke a real frontier
model for that behavior.

**Why it matters for jailbreaking**
These are scientifically validated jailbreaks with documented success rates against the most
heavily defended models. If a prompt structure broke GPT-4, a version of it will likely break a
smaller local model with far weaker safety training.

**When Gemini uses it**
Called when a proven starting-point prompt is wanted rather than generating from scratch, or
when standard techniques keep failing and a tested structure is needed.

---

### Tool 6 — `search_web` · DuckDuckGo Live Search

**What it does**
Performs a real-time DuckDuckGo web search using the [`duckduckgo-search`](https://github.com/deedy5/duckduckgo_search)
Python library (no API key required) and returns live results including titles, URLs, and snippets.

**Why it matters for jailbreaking**
New bypasses for specific model versions appear on Reddit, GitHub, and HuggingFace forums
constantly. Gemini's training data has a cutoff date — the web has today's exploits.

**When Gemini uses it**
Called when standard and tool-guided techniques have all failed, and a recently published
model-specific exploit may exist that Gemini's training data does not cover.

---

## How Gemini Triggers Tools and Receives Results

Understanding this mechanism explains why tool output persists across every attack attempt.

### 1 — Tool declarations (what Gemini "sees")

Gemini never imports or reads `tools.py`. Instead, at session startup the backend sends Gemini a
list of **function declarations** — typed schemas describing each tool's name, description, and
parameters:

```python
# backend/gemini.py
JAILBREAK_TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(name="get_model_profile", ...),
        types.FunctionDeclaration(name="run_garak_probe", ...),
        types.FunctionDeclaration(name="apply_pyrit_converter", ...),
        types.FunctionDeclaration(name="run_pyrit_crescendo", ...),
        types.FunctionDeclaration(name="fetch_jailbreak_prompts", ...),
        types.FunctionDeclaration(name="search_web", ...),
    ]
)
```

These declarations are passed in the `GenerateContentConfig` attached to every Gemini call in
the session. Gemini reads them like a typed API contract and decides on its own when and how to
call each tool.

### 2 — Gemini responds with a function call instead of text

When Gemini decides to use a tool, the API response does not contain text. Instead it contains a
structured **function call object**:

```json
{ "name": "run_garak_probe", "args": { "target_model": "llama3.2", "probe_name": "dan" } }
```

The backend detects this with `_get_function_calls(response)`, which checks
`response.function_calls` or walks `response.candidates[0].content.parts` looking for
`function_call` parts.

### 3 — The backend executes the real tool

```python
result = attack_tools.execute_tool(fc.name, dict(fc.args))
```

This calls the actual Python function in `tools.py` — running a real `subprocess` for Garak or
PyRIT, a real `httpx.get()` for JailbreakBench, a real DuckDuckGo search, or a real Ollama API
call. The function returns a plain string with the full output.

### 4 — The output is sent back to Gemini as a `FunctionResponse`

The result string is wrapped in a `types.FunctionResponse` and appended to the conversation
history as a new message turn:

```python
types.Part(
    function_response=types.FunctionResponse(
        name=fc.name,
        response={"result": result},   # ← full tool output
    )
)
```

Gemini's next API call receives the entire updated history, so it reads the tool output exactly
like a message in a chat. It can then call another tool or write its attack prompt.

### 5 — The loop repeats until Gemini writes text

```
for _ in range(MAX_TOOL_ROUNDS + 1):   # up to 8 rounds
    response = Gemini(working)
    if no function calls in response:
        break   ← Gemini wrote a text response, done
    execute tools → append FunctionResponse → loop again
```

In a single `send()` call, Gemini may call several tools in sequence before producing the attack
prompt. Each iteration the history list grows.

### 6 — All tool results persist for the entire attack session

`GeminiAttackSession` keeps a single `self._contents` list that grows across every attack attempt:

```
Attack attempt 1:
  user msg → Gemini calls get_model_profile → result → Gemini calls run_garak_probe("dan")
  → result → Gemini writes prompt #1
  self._contents = [all of the above]

Attack attempt 2:
  working = list(self._contents)   ← includes profile and Garak results from attempt 1
  + new user msg → Gemini writes prompt #2 (no re-running tools it already has)
  self._contents = [everything so far + new round]
```

Gemini does not need to re-run any tool to remember what it found earlier — every tool result is
literally in the message history it reads at the start of every call. A Garak scan from attempt 1
is still in context in attempt 7.

### Strategic tool decision

The system prompt tells Gemini the exact condition under which each tool adds value and when to
skip it. Gemini calls tools only when they provide genuinely new information — not as a ritual on
every attempt. This keeps attack sessions fast and focused.

```
get_model_profile    → call if template format or family weaknesses are unknown
fetch_jailbreak_prompts → call if a proven starting prompt is wanted
run_garak_probe      → call before committing to an encoding or technique category
apply_pyrit_converter → call when keyword filters keep catching a good prompt
run_pyrit_crescendo  → call when single-turn attacks keep failing
search_web           → call when all else fails and a recent exploit may be indexed
```

---

## Run with Docker Compose (recommended)

Docker Compose starts all three services — Ollama, backend, and frontend — with a single command.

### 1) Create the environment file

```bash
cp .env.example .env
```

Open `.env` and set your Gemini API key:

```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 2) Build and start all services

```bash
docker compose up --build
```

### How the startup sequence works

Docker Compose starts services in a strict order:

```
1. ollama        → starts the local LLM server
2. ollama-init   → waits for ollama, then downloads all target models
3. backend       → waits for ollama-init to finish, then starts the API
4. frontend      → waits for backend to be healthy, then opens the UI
```

**First run** downloads the Ollama image, builds the Python images, and pulls all models.
This can take **5–20 minutes** depending on your internet speed (models are 1–4 GB each).

**Subsequent runs** skip the downloads — models are cached in the `ollama_data` Docker volume
and startup takes only a few seconds.

To watch model download progress in real time:

```bash
docker compose logs -f ollama-init
```

### 3) Open the app

| Service    | URL                             |
|------------|---------------------------------|
| Frontend   | http://localhost:8501           |
| Statistics | http://localhost:8501/stats     |
| Backend    | http://localhost:8000           |
| Swagger    | http://localhost:8000/docs      |

### Add more target models

```bash
docker compose exec ollama ollama pull <model-name>
```

Browse available models at [ollama.com/library](https://ollama.com/library).

### Stop all services

```bash
docker compose down
```

> **Data persistence:** Session history (`db_data` volume) and downloaded models (`ollama_data` volume) survive restarts.
> To wipe everything and start fresh: `docker compose down -v`

---

## Manual Setup

Run from the project root (`AgentXploit/`).

### 1) Create and activate virtual environment

#### Windows (PowerShell)
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

#### macOS/Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2) Install dependencies

```bash
pip install -r requirements.txt
```

### 3) Create environment file

Create `backend/.env`:

```env
GEMINI_API_KEY=your_actual_api_key_here
```

### 4) Run (two terminals)

Start Ollama first:

```bash
ollama serve
```

**Terminal 1 — Backend:**
```bash
cd backend
uvicorn main:app --port 8000 --reload
```

**Terminal 2 — Frontend:**
```bash
cd frontend
streamlit run app.py
```

---

## Common Issues

**Docker**
- Frontend not loading yet: models are still downloading — run `docker compose logs -f ollama-init`.
- Port conflict (`8000` or `8501`): stop the other process or change the host port in `docker-compose.yml`.
- `GEMINI_API_KEY` not set: create `.env` from `.env.example` before running `docker compose up`.

**Manual / general**
- `503 UNAVAILABLE` from Gemini: temporary service overload; the backend retries automatically.
- `Connection refused` for `127.0.0.1:8000`: backend is not running.
- Empty models list: Ollama is not running or no local models are installed.
