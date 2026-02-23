# Domain Q&A Chatbot – Introductory Data Analytics

**Repository:** https://github.com/g7yue/domain-lite-chatbot  
**Live URL:** https://domain-lite-chatbot-32zzvwbxsq-uc.a.run.app

Build a domain-restricted web chatbot that answers questions strictly within Introductory Data Analytics. The assistant responds to beginner-level analytics questions and safely refuses out-of-scope requests.


## Domain Scope

The chatbot answers questions about:

* Descriptive statistics (mean, median, variance, standard deviation, correlation)
* Business metrics (conversion rate, retention rate, average order value)
* High-level SQL or pandas analysis concepts (no full scripts)
* Data analysis workflows (EDA, hypothesis testing concepts)
* Data quality topics (missing values, sampling bias)

The chatbot does **not** answer questions outside this scope.
Because the domain is restricted to introductory analytics, it will refuse questions involving:

* Machine learning model training  
* Neural networks or deep learning  
* AI model architecture or tuning  
* Medical or clinical advice  
* Legal or compliance topics  
* Software engineering or backend system design  

Out-of-scope queries return exactly:

`This question is outside of my analytics domain.`


## Run Locally

1. Install dependencies and run the app:

```bash
uv run python app.py
```

2. Open [http://127.0.0.1:8080](http://127.0.0.1:8080) in your browser.

3. Run the evaluation harness (single command per project requirement):

```bash
uv run python eval.py
```


## How the Model Is Used

The app uses **TinyLlama 1.1B Chat** via the HuggingFace Inference API for novel in-domain questions, with rules and caches so common cases are fast and scope is enforced. No model is downloaded — the container makes HTTP calls to HuggingFace's hosted endpoint.

* **When the model is called**
  Only when the user message is in-domain and not handled by the rules below. The prompt uses the ChatML format and includes: **role/persona** (introductory analytics assistant), **positive constraints** (topics the bot can answer), **≥3 few-shot examples** (mean, variance, correlation + one out-of-scope refusal), and an **escape hatch** (respond exactly with the refusal phrase when unsure or out-of-scope).

* **Before the model**
  Response cache → instant reply for repeat questions. Exact matches to 15 canonical in-domain questions → pre-defined answers (no API call). Safety keywords → fixed gentle signpost. Out-of-scope keywords (regex) → refusal phrase. Greetings → fixed welcome message.

* **After the model (Python backstop)**
  A regex + keyword backstop runs on the generated text; if it detects out-of-scope or safety content, the reply is replaced with the refusal or safety message (defense-in-depth).

* **Loading**
  No model weights in the container. The HuggingFace Inference API (featherless-ai provider) is called at request time via HTTP. An `HF_TOKEN` is required — for local dev, copy `.env.example` to `.env` and fill in your token; for Cloud Run, the token is already configured as a service environment variable.


## What's Included

* `app.py` – FastAPI backend with TinyLlama via HuggingFace Inference API
* `index.html` – Web frontend  
* `eval.py` – Automated evaluation harness  
* Structured prompt with:  
  * Role + persona  
  * Positive domain constraints  
  * 3+ few-shot examples  
  * Explicit out-of-scope categories  
  * Escape hatch  
* Regex-based Python backstop filter
* TinyLlama 1.1B Chat via HuggingFace Inference API.


## Evaluation

The evaluation harness meets the project requirements: a single command runs all tests and reports results.

* **Dataset:** 20+ cases — 15 in-domain (with expected answers), 5 out-of-scope, 5 adversarial/safety-trigger.  
* **Metrics:** ≥1 deterministic metric (refusal detection: exact match to the refusal phrase); golden-reference (F1-style overlap with expected answer); rubric (keyword-weighted scoring).  
* **Output:** Pass/fail per test, pass rates by category (in_domain, out_of_scope, adversarial), and overall pass rate.

Run: `uv run python eval.py` (see **Run Locally** above).


## Additional Engineering

Beyond the core project requirements, the following robustness improvements were implemented:

* **Truncated response guard** — After model generation, if the response ends mid-sentence (i.e. the last character is not `.`, `?`, or `!`), the text is trimmed to the last complete sentence. This prevents garbled half-answers from reaching the user when the model runs out of tokens mid-generation.

* **In-memory response cache** — A 500-entry cache (keyed on the normalized question) stores question→answer pairs. Repeated questions are served instantly without an API call.

* **Canonical answer fast-path** — 15 common introductory analytics questions are mapped to pre-defined, textbook-quality answers. These bypass the model entirely for both speed and consistency.

* **Multi-layer input pipeline** — Each message passes through five checks before reaching the model: (1) cache lookup, (2) canonical answer match, (3) safety keyword filter, (4) out-of-scope regex filter, (5) greeting detection. The model is only called when all five checks pass.

* **Graceful API fallback** — If the HuggingFace API call fails for any reason (timeout, rate limit, server error), the user receives an informative fallback message instead of a crash or empty response.

* **HF_TOKEN configured end-to-end** — Token is loaded from `.env` for local development (via `python-dotenv`) and set as a Cloud Run environment variable for production, so the app runs correctly in both environments without code changes.
