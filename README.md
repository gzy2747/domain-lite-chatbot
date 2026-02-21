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

The app uses **TinyLlama 1.1B Chat** for in-domain answers, with rules and caches so common cases are fast and scope is enforced.

* **When the model is called**  
  Only when the user message is in-domain and not handled by rules below. The prompt includes: **role/persona** (introductory analytics assistant), **positive constraints** (topics the bot can answer), **≥3 few-shot examples** (e.g. mean, variance, correlation + one out-of-scope refusal), and an **escape hatch** (respond exactly with the refusal phrase when unsure or out-of-scope).

* **Before the model**  
  Greetings (e.g. “hi”, “how are you”) → fixed welcome message. Safety-related keywords → fixed gentle signpost. Out-of-scope keywords (regex) → refusal phrase. Exact matches to the 15 canonical in-domain questions → pre-defined answers (no model call). Response cache → repeat questions return the same answer instantly.

* **After the model**  
  Output is trimmed to the last complete sentence if the token limit cuts the reply mid-sentence. A **Python backstop** (regex + safety keywords) runs on the generated text; if it detects out-of-scope or safety content, the reply is replaced with the refusal or safety message (defense-in-depth).

* **Loading**  
  The model is lazy-loaded on first use and preloaded in a background thread at startup so the server binds to the port immediately (for Cloud Run) and the first request is faster when possible.


## What's Included

* `app.py` – FastAPI backend with session management and TinyLlama-based response generation  
* `index.html` – Web frontend  
* `eval.py` – Automated evaluation harness  
* Structured prompt with:  
  * Role + persona  
  * Positive domain constraints  
  * 3+ few-shot examples  
  * Explicit out-of-scope categories  
  * Escape hatch  
* Regex-based Python backstop filter (defense-in-depth)  
* Local open-source model backend (TinyLlama 1.1B Chat) — **no external API required**  


## Evaluation

The evaluation harness meets the project requirements: a single command runs all tests and reports results.

* **Dataset:** 20+ cases — 15 in-domain (with expected answers), 5 out-of-scope, 5 adversarial/safety-trigger.  
* **Metrics:** ≥1 deterministic metric (refusal detection: exact match to the refusal phrase); golden-reference (F1-style overlap with expected answer); rubric (keyword-weighted scoring).  
* **Output:** Pass/fail per test, pass rates by category (in_domain, out_of_scope, adversarial), and overall pass rate.

Run: `uv run python eval.py` (see **Run Locally** above).
