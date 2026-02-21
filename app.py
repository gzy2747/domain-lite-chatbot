import logging
import re
import threading
import uuid
import uvicorn
import torch

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM


# MODEL (lazy-loaded so Cloud Run sees the container listen on PORT before timeout)
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
_tokenizer = None
_model = None
_model_lock = threading.Lock()


def _get_model():
    global _tokenizer, _model
    with _model_lock:
        if _model is None:
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            _model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            _model.to(device)
            _model.eval()
        return _tokenizer, _model


def _preload_model():
    """Run in background so model is ready sooner; container still binds to PORT immediately."""
    _get_model()



# Regex Backstop
OUT_OF_SCOPE_REGEX = re.compile(
    r"(deep learning|neural networks?|machine learning|ai models?|"
    r"doctor|medical|therapy|diagnosis|treatment|"
    r"legal|lawyer|compliance|regulation|"
    r"software architecture|system design|backend)",
    re.IGNORECASE,
)

# Safety backstop: distressed keywords → fixed gentle response (escape hatch)
DISTRESSED_REGEX = re.compile(
    r"(suicide|kill myself|hurt myself|self[\s-]?harm|end (it|my life)|"
    r"want to die|crisis line|helpline)",
    re.IGNORECASE,
)


def is_out_of_scope(text: str) -> bool:
    return bool(OUT_OF_SCOPE_REGEX.search(text))


def is_safety_trigger(text: str) -> bool:
    return bool(DISTRESSED_REGEX.search(text))


# Greetings → fixed friendly reply (no model call)
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
    # Very short and looks like a greeting
    words = t.lower().split()
    if len(words) <= 3 and any(w in ("hi", "hello", "hey", "yo") for w in words):
        return True
    return False


def _trim_to_last_sentence(text: str) -> str:
    """If text was cut off by token limit, trim back to last complete sentence."""
    text = text.strip()
    if not text or text[-1] in ".!?":
        return text
    # Find last sentence terminator and cut there
    for sep in (". ", "! ", "? "):
        idx = text.rfind(sep)
        if idx != -1:
            return text[: idx + 1].strip()
    for sep in (".", "!", "?"):
        idx = text.rfind(sep)
        if idx != -1:
            return text[: idx + 1].strip()
    # No sentence end: trim at last comma to avoid mid-phrase
    idx = text.rfind(", ")
    if idx != -1:
        return text[: idx + 1].strip()
    return text


# Response cache (same question → instant reply)
_response_cache: dict[str, str] = {}
_CACHE_MAX = 500


# Instant answers for exact questions you defined (no model call = no wait)
def _normalize(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


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


# Generator
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

    chatml_prompt = f"""
<|system|>
You are an Introductory Data Analytics assistant.

You only answer questions within these topics:
- Descriptive statistics (mean, median, variance, correlation)
- Basic business metrics (conversion rate, retention rate)
- Hypothesis testing fundamentals
- Exploratory data analysis
- Data quality (missing values, sampling bias) and EDA

Provide textbook-style definitions.
Start with "[Term] is" or "[Term] measures".
Avoid paraphrasing core statistical terms.

For any question outside the topics above, respond exactly with:
"This question is outside of my analytics domain."

<|user|>
What is mean?
<|assistant|>
The mean is the average value obtained by summing all observations and dividing by the number of observations.

<|user|>
What is variance?
<|assistant|>
Variance measures how far values are spread from the mean.

<|user|>
What is correlation?
<|assistant|>
Correlation measures the strength and direction of the relationship between two variables.

<|user|>
Explain deep learning.
<|assistant|>
This question is outside of my analytics domain.

<|user|>
{question}
<|assistant|>

""".strip()

    tokenizer, model = _get_model()
    device = next(model.parameters()).device
    inputs = tokenizer(chatml_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=False)

    generated_text = decoded[len(chatml_prompt):]

    stop_tokens = ["<|user|>", "<|assistant|>"]

    answer = generated_text
    for token in stop_tokens:
        if token in answer:
            answer = answer.split(token)[0]

    answer = re.sub(r"<\|/?[a-z]+\|>", "", answer)
    # Clean model output artifact (leftover token fragments)
    answer = answer.replace("nt|>", "").strip()
    # Trim to last complete sentence so we never end mid-phrase when tokens run out
    answer = _trim_to_last_sentence(answer)

    if is_safety_trigger(answer):
        out = "I'm not able to help with that. If you're going through a difficult time, please reach out to a mental health professional or a crisis line in your area."
        if len(_response_cache) < _CACHE_MAX:
            _response_cache[key] = out
        return out

    if is_out_of_scope(answer):
        answer = "This question is outside of my analytics domain."

    if len(_response_cache) < _CACHE_MAX:
        _response_cache[key] = answer
    return answer




# FastAPI
app = FastAPI()

# Start loading model in background so first request is faster (container still binds to PORT immediately)
threading.Thread(target=_preload_model, daemon=True).start()


@app.exception_handler(Exception)
def unhandled_exception_handler(request: Request, exc: Exception):
    logging.exception("unhandled error")
    return JSONResponse(
        status_code=200,
        content={
            "response": "Sorry, something went wrong. Please try again.",
            "session_id": str(uuid.uuid4()),
        },
    )


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
        return ChatResponse(
            response="Sorry, something went wrong. Please try again.",
            session_id=str(uuid.uuid4()),
        )


if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
