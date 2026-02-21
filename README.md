# Domain Q&A Chatbot – Introductory Data Analytics

**Repository:** https://github.com/g7yue/domain-lite-chatbot  
**Live URL:**

Build a domain-restricted web chatbot that answers questions strictly within Introductory Data Analytics.

The assistant responds to beginner-level analytics questions and safely refuses out-of-scope requests.


## Domain Scope

The chatbot answers questions about:

* Descriptive statistics (mean, median, variance, standard deviation, correlation)
* Business metrics (conversion rate, retention rate, average order value)
* High-level SQL or pandas analysis concepts (no full scripts)
* Data analysis workflows (EDA, hypothesis testing concepts)
* Data quality topics (missing values, sampling bias)

The chatbot does **not** answer questions outside this scope.
Because the domain is restricted to introductory analytics, it will **refuse** questions involving:

* Machine learning model training  
* Neural networks or deep learning  
* AI model architecture or tuning  
* Medical or clinical advice  
* Legal or compliance topics  
* Software engineering or backend system design  

Out-of-scope queries return exactly:

"This question is outside my analytics domain."


## Run Locally

1. Install dependencies and run the app:

```bash
uv run python app.py
```

2. Open [http://127.0.0.1:8080](http://127.0.0.1:8080) in your browser.

3. Run the evaluation harness (single command):

```bash
uv run python eval.py
```


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

The project includes an automated evaluation script that satisfies the project requirements.

The evaluation includes:

* 20+ test cases  
* In-domain questions  
* Out-of-scope questions  
* Adversarial prompts  
* Deterministic refusal detection  
* Golden-reference (semantic) scoring  
* Rubric-based scoring  
* Pass/fail per test  
* Category pass rates  
* Overall pass rate  

Run evaluation with: `uv run python eval.py` (see **Run Locally** above).
