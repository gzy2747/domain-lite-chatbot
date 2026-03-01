# Domain Q&A Chatbot – Cat Behavior

**Repository:** https://github.com/g7yue/domain-lite-chatbot

**Live URL:** https://domain-lite-chatbot-521802278218.us-central1.run.app

Build a domain-restricted web chatbot that answers questions strictly within cat behavior. The assistant responds to questions about why cats do what they do, and safely refuses out-of-scope requests.


## Domain Scope

The chatbot answers questions about cat behavior including:

* Why cats exhibit specific behaviors (kneading, purring, scratching, chirping, etc.)
* Cat communication and body language (slow blink, headbutt, tail positions, etc.)
* Cat instincts and natural habits (hunting, sleeping, grooming, etc.)
* Cat-human bonding behaviors
* Common cat quirks and what they mean

The chatbot does **not** answer questions outside this scope.
It will refuse questions involving:

* Machine learning or AI models
* Stocks, crypto, or trading
* Cooking recipes
* Legal or compliance topics
* Medical advice
* Dog training or other non-cat topics
* Software engineering or backend system design
* Statistics or data analytics

Out-of-scope queries return exactly:

`This question is outside of my cat behavior domain.`


## Run Locally

**Option A — No HF token (local model, recommended for graders):**

First install the local-inference extras (downloads ~2 GB of PyTorch + Transformers once):

```bash
uv sync --extra local
uv run python app.py
```

The TinyLlama weights (~2.2 GB) are downloaded automatically on the first request and cached by HuggingFace. Subsequent runs load from cache instantly.

**Option B — HF token (API, faster startup):**

```bash
cp .env.example .env   # fill in HF_TOKEN
uv run python app.py
```

Either way, open [http://127.0.0.1:8080](http://127.0.0.1:8080) in your browser.

Run the evaluation harness (single command per project requirement):

```bash
uv run python eval.py
```


## How the Model Is Used

The app uses **TinyLlama 1.1B Chat** for novel in-domain questions, with rules and caches so common cases are fast and scope is enforced. Two inference modes are supported automatically:

* **API mode** (when `HF_TOKEN` is set): makes HTTP calls to the HuggingFace Inference API (featherless-ai provider). No model weights in the container — used by Cloud Run.
* **Local mode** (when `HF_TOKEN` is absent): loads TinyLlama weights locally via the `transformers` library. Weights are downloaded once (~2.2 GB) and cached by HuggingFace. No token required.

* **When the model is called**
  Only when the user message is in-domain and not handled by the rules below. The prompt uses the ChatML format and includes: **role/persona** (cat behavior expert assistant), **positive constraints** (topics the bot can answer), **≥3 few-shot examples** (knead, bring dead animals, purr + one out-of-scope refusal), and an **escape hatch** (respond exactly with the refusal phrase when unsure or out-of-scope).

* **Before the model**
  Response cache → instant reply for repeat questions. Exact matches to 15 canonical cat behavior questions → pre-defined answers (no model call). Safety keywords → fixed gentle signpost. Out-of-scope keywords (regex) → refusal phrase. Food safety questions → vet disclaimer. Greetings → fixed welcome message.

* **After the model (Python backstop)**
  A regex + keyword backstop runs on the generated text; if it detects out-of-scope or safety content, the reply is replaced with the refusal or safety message (defense-in-depth).


## What's Included

* `app.py` – FastAPI backend with TinyLlama (local via `transformers` or HF Inference API, auto-detected)
* `index2.html` – Web frontend (cat behavior UI)
* `eval.py` – Automated evaluation harness
* Structured prompt with:
  * Role + persona
  * Positive domain constraints
  * 3+ few-shot examples
  * Explicit out-of-scope categories
  * Escape hatch
* Regex-based Python backstop filter
* TinyLlama 1.1B Chat — local inference (`transformers`) or HuggingFace Inference API, auto-detected.


## Evaluation

The evaluation harness meets the project requirements: a single command runs all tests and reports results.

* **Dataset:** 30 cases — 20 in-domain (15 canonical + 5 novel), 5 out-of-scope, 5 adversarial.
* **Metrics:** ≥1 deterministic metric (refusal detection: exact match to the refusal phrase); golden-reference (F1-style overlap with expected answer); rubric (keyword-weighted scoring).
* **Output:** Pass/fail per test, pass rates by category (in_domain, out_of_scope, adversarial), and overall pass rate.

Run: `uv run python eval.py` (see **Run Locally** above).


## Additional Engineering

Beyond the core project requirements, the following robustness improvements were implemented:

* **Truncated response guard** — After model generation, two checks run in sequence. First, if the response ends mid-sentence (last character is not `.`, `?`, or `!`), the text is trimmed to the last complete sentence. Second, if the model hit the token limit mid-list and left a dangling numbered item with no content (e.g. `\n\n5.`), that empty item is stripped via regex. Together these prevent garbled or incomplete answers from reaching the user.

* **In-memory response cache** — A 500-entry cache (keyed on the normalized question) stores question→answer pairs. Repeated questions are served instantly without an API call.

* **Canonical answer fast-path** — 15 common cat behavior questions are mapped to pre-defined, expert-quality answers. These bypass the model entirely for both speed and consistency.

* **Multi-layer input pipeline** — Each message passes through six checks before reaching the model: (1) cache lookup, (2) canonical answer match, (3) safety keyword filter, (4) out-of-scope regex filter, (5) food safety filter, (6) greeting detection. The model is only called when all six checks pass.

* **Food safety disclaimer** — TinyLlama (1.1B parameters) is too small to be reliably accurate on pet nutrition facts. Questions about what cats can or cannot eat are intercepted before reaching the model and receive a fixed response directing users to consult their veterinarian. This prevents the model from producing plausible-sounding but factually incorrect food safety advice.

* **Dual inference mode** — The app auto-detects whether an `HF_TOKEN` is present. When the token is set (Cloud Run), it uses the HuggingFace Inference API for fast, lightweight responses. When no token is present (local clone), it loads TinyLlama via the `transformers` library with no token required. Local deps are declared as an optional group (`uv sync --extra local`) so the Cloud Run container stays small.

* **Graceful fallback** — If the model call fails for any reason (API timeout, rate limit, local OOM), the user receives an informative fallback message instead of a crash or empty response.

* **Adaptive suggestion chips** — On the welcome screen, quick-question chips are displayed front-and-center to guide new users. Once the first message is sent and the welcome view is dismissed, the same chips reappear as a compact scrollable bar pinned above the input box, so suggested questions remain accessible throughout the conversation without cluttering the chat history.