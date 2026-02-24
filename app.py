import logging
import re
import uuid
import uvicorn
import os
import traceback

import requests
from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel


# HuggingFace Inference via featherless-ai provider (OpenAI-compatible chat format).
# Requires HF_TOKEN with Inference Providers permission (free HF account).
_HF_API_URL = "https://router.huggingface.co/featherless-ai/v1/chat/completions"
_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

# Structured prompt: role/persona + positive constraints + ≥3 few-shot examples + escape hatch
_SYSTEM_PROMPT = """You are an Introductory Data Analytics assistant.

You can answer questions about:
- Descriptive statistics (mean, median, mode, variance, standard deviation, correlation)
- Basic business metrics (conversion rate, retention rate, average order value)
- Hypothesis testing fundamentals (null hypothesis, p-value, significance)
- Exploratory data analysis (EDA, distributions, outliers)
- Data quality (missing values, sampling bias)

Provide strict textbook-style definitions only.

Rules:
- Give exactly 1–2 sentences.
- Define the concept directly.
- Do not provide examples.
- Do not provide step-by-step explanations.
- Do not mention specific statistical distributions unless explicitly asked.
- Do not add background context.
- Do not speculate.

When you are unsure or the question is outside these topics, respond exactly with:
"This question is outside of my analytics domain."
"""

_FEW_SHOT = [
    {"role": "user",      "content": "What is mean?"},
    {"role": "assistant", "content": "The mean is the average value obtained by summing all observations and dividing by the number of observations."},
    {"role": "user",      "content": "What is variance?"},
    {"role": "assistant", "content": "Variance measures how far values are spread from the mean."},
    {"role": "user",      "content": "What is correlation?"},
    {"role": "assistant", "content": "Correlation measures the strength and direction of the relationship between two variables."},
    {"role": "user",      "content": "Explain deep learning."},
    {"role": "assistant", "content": "This question is outside of my analytics domain."},
]


def _call_hf(question: str) -> str:
    token = os.environ.get("HF_TOKEN", "")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": _MODEL,
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            *_FEW_SHOT,
            {"role": "user", "content": question},
        ],
        "max_tokens": 150,
        "temperature": 0.1,
    }
    try:
        resp = requests.post(_HF_API_URL, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception:
        logging.exception("HF API call failed")
        return _FALLBACK


# Regex Backstop (defense-in-depth after generation)
OUT_OF_SCOPE_REGEX = re.compile(
    r"(deep learning|neural networks?|machine learning|ai models?|"
    r"doctor|medical|therapy|diagnosis|treatment|"
    r"legal|lawyer|compliance|regulation|"
    r"software architecture|system design|backend)",
    re.IGNORECASE,
)

DISTRESSED_REGEX = re.compile(
    r"(suicide|kill myself|hurt myself|self[\s-]?harm|end (it|my life)|"
    r"want to die|crisis line|helpline)",
    re.IGNORECASE,
)


def is_out_of_scope(text: str) -> bool:
    return bool(OUT_OF_SCOPE_REGEX.search(text))


def is_safety_trigger(text: str) -> bool:
    return bool(DISTRESSED_REGEX.search(text))


GREETING_PATTERN = re.compile(
    r"^(hi|hello|hey|howdy|yo|greetings?|good\s*(morning|afternoon|evening)|"
    r"how\s*are\s*(you|u)\??\s*$|what'?s\s*up\??\s*$|hi\s+there\s*$)",
    re.IGNORECASE,
)


def is_greeting(text: str) -> bool:
    t = text.strip()
    if not t or len(t) > 80:
        return False
    if GREETING_PATTERN.match(t):
        return True
    words = t.lower().split()
    if len(words) <= 3 and any(w in ("hi", "hello", "hey", "yo") for w in words):
        return True
    return False


# Response cache
_response_cache: dict[str, str] = {}
_CACHE_MAX = 500


def _normalize(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower()).rstrip("?.!")


CANONICAL_ANSWERS = {
    _normalize("What is mean?"): "The mean is the average value obtained by summing all observations and dividing by the number of observations.",
    _normalize("What is median?"): "The median is the middle value in a dataset after the values are arranged in order.",
    _normalize("What is variance?"): "Variance measures how far values are spread from the mean.",
    _normalize("What is standard deviation?"): "Standard deviation measures how dispersed values are relative to the mean.",
    _normalize("What is correlation?"): "Correlation measures the strength and direction of the relationship between two variables.",
    _normalize("What is conversion rate?"): "Conversion rate is the percentage of users who complete a desired action.",
    _normalize("What is retention rate?"): "Retention rate measures the percentage of users who return over a specific period.",
    _normalize("What is average order value?"): "Average order value is the average revenue generated per order.",
    _normalize("What is hypothesis testing?"): "Hypothesis testing is a statistical method used to determine whether there is enough evidence to reject a null hypothesis.",
    _normalize("What is a null hypothesis?"): "A null hypothesis is a default assumption that there is no effect or relationship.",
    _normalize("What is sampling bias?"): "Sampling bias occurs when a sample is not representative of the population.",
    _normalize("What is exploratory data analysis?"): "Exploratory data analysis (EDA) is the process of analyzing data to identify patterns and relationships.",
    _normalize("Why check for missing values?"): "Missing values can bias results and should be identified before performing analysis.",
    _normalize("What does correlation coefficient indicate?"): "The correlation coefficient indicates the strength and direction of a linear relationship between variables.",
    _normalize("Why is data visualization important?"): "Data visualization helps identify patterns, trends, and outliers in data.",
}

_FALLBACK = (
    "I can answer introductory data analytics questions — for example: "
    "mean, median, variance, standard deviation, correlation, hypothesis testing, "
    "sampling bias, exploratory data analysis, conversion rate, or retention rate."
)


def generate_response(question: str) -> str:
    key = _normalize(question)
    if key in _response_cache:
        return _response_cache[key]

    if key in CANONICAL_ANSWERS:
        return CANONICAL_ANSWERS[key]

    if is_safety_trigger(question):
        out = "I'm not able to help with that. If you're going through a difficult time, please reach out to a mental health professional or a crisis line in your area."
        if len(_response_cache) < _CACHE_MAX:
            _response_cache[key] = out
        return out

    if is_out_of_scope(question):
        out = "This question is outside of my analytics domain."
        if len(_response_cache) < _CACHE_MAX:
            _response_cache[key] = out
        return out

    if is_greeting(question):
        out = "Hi! I'm an introductory data analytics bot. Feel free to ask me about things like mean, correlation, hypothesis testing, or other basic analytics topics."
        if len(_response_cache) < _CACHE_MAX:
            _response_cache[key] = out
        return out

    # Call the model for novel in-domain questions
    answer = _call_hf(question)

    # Python backstop: re-check generated output (defense-in-depth)
    if is_safety_trigger(answer):
        answer = "I'm not able to help with that. If you're going through a difficult time, please reach out to a mental health professional or a crisis line in your area."
    elif is_out_of_scope(answer):
        answer = "This question is outside of my analytics domain."
    else:
        # Truncation guard: if the model ran out of tokens mid-sentence,
        # trim to the last complete sentence rather than returning a cut-off reply.
        if answer and answer[-1] not in ".?!":
            last_end = max(answer.rfind("."), answer.rfind("?"), answer.rfind("!"))
            if last_end > 0:
                answer = answer[:last_end + 1]

    if len(_response_cache) < _CACHE_MAX:
        _response_cache[key] = answer
    return answer


# FastAPI
app = FastAPI()

DEBUG = os.environ.get("DEBUG", "0") in ("1", "true", "True")


@app.exception_handler(Exception)
def unhandled_exception_handler(request: Request, exc: Exception):
    logging.exception("unhandled error")
    content = {
        "response": "Sorry, something went wrong. Please try again.",
        "session_id": str(uuid.uuid4()),
    }
    if DEBUG:
        content["error"] = str(exc)
        content["traceback"] = traceback.format_exc()
    return JSONResponse(status_code=500, content=content)


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str
    session_id: str


@app.get("/")
def index():
    return FileResponse("index.html")


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    try:
        response = generate_response(request.message)
        return ChatResponse(
            response=response,
            session_id=str(uuid.uuid4()),
        )
    except Exception as e:
        logging.exception("chat error")
        if DEBUG:
            resp_text = f"Sorry, something went wrong. {e}"
        else:
            resp_text = "Sorry, something went wrong. Please try again."
        return ChatResponse(
            response=resp_text,
            session_id=str(uuid.uuid4()),
        )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
