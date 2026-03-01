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
_SYSTEM_PROMPT = """You are a cat behavior expert assistant.

You can only answer questions about:
- Why cats exhibit specific behaviors (kneading, purring, scratching, chirping, etc.)
- Cat communication and body language (slow blink, headbutt, tail positions, etc.)
- Cat instincts and natural habits (hunting, sleeping, grooming, etc.)
- Cat-human bonding behaviors
- Common cat quirks and what they mean

Provide clear, factual explanations in 1-3 sentences.

When you are unsure or the question is outside these topics, respond exactly with:
"This question is outside of my cat behavior domain."
"""

_FEW_SHOT = [
    {"role": "user",      "content": "Why do cats knead?"},
    {"role": "assistant", "content": "Cats knead as a comforting behavior from kittenhood nursing, and continue it as adults when feeling safe and content."},
    {"role": "user",      "content": "Why do cats bring dead animals?"},
    {"role": "assistant", "content": "Cats bring dead prey as a gift rooted in their hunting instinct, treating their owners as part of their family group to provide for."},
    {"role": "user",      "content": "Why do cats purr?"},
    {"role": "assistant", "content": "Cats purr to express contentment, but also when stressed or injured, as the vibration frequency may promote physical healing."},
    {"role": "user",      "content": "What is the stock market?"},
    {"role": "assistant", "content": "This question is outside of my cat behavior domain."},
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
    r"stocks?|crypto(?:currency)?|bitcoin|trading|investment|"
    r"recipe|how to cook|baking|chef|"
    r"legal|lawyer|compliance|regulation|"
    r"medical advice|prescri|"
    r"dog training|train a dog|"
    r"software architecture|system design|backend|"
    r"statistics|statistical|median|variance|standard deviation|"
    r"correlation|regression|hypothesis test|data analytics|"
    r"what is (a |the )?mean\b|what'?s (a |the )?mean\b)",
    re.IGNORECASE,
)

DISTRESSED_REGEX = re.compile(
    r"(suicide|kill myself|hurt myself|self[\s-]?harm|end (it|my life)|"
    r"want to die|crisis line|helpline)",
    re.IGNORECASE,
)

# Food safety questions — TinyLlama is too small to be reliably accurate on
# pet nutrition; deflect all such questions to a veterinarian disclaimer.
FOOD_SAFETY_REGEX = re.compile(
    r"(can cats? eat|safe for cats? to eat|is .{0,30} safe for cats?|"
    r"can i feed (my )?cats?|is .{0,30} toxic to cats?|"
    r"can cats? (drink|have|consume))",
    re.IGNORECASE,
)


def is_out_of_scope(text: str) -> bool:
    return bool(OUT_OF_SCOPE_REGEX.search(text))


def is_safety_trigger(text: str) -> bool:
    return bool(DISTRESSED_REGEX.search(text))


def is_food_safety(text: str) -> bool:
    return bool(FOOD_SAFETY_REGEX.search(text))


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
    _normalize("Why do cats knead?"): "Cats knead as a comforting behavior from kittenhood nursing, and continue it as adults when feeling safe and content.",
    _normalize("Why do cats purr?"): "Cats purr to express contentment, but also when stressed or injured, as the vibration frequency may promote physical healing.",
    _normalize("Why do cats rub against people?"): "Cats rub against people to deposit scent from their facial glands, marking them as part of their territory and showing affection.",
    _normalize("Why do cats push things off tables?"): "Cats push objects off surfaces out of curiosity, to test if they move, and to attract their owner's attention.",
    _normalize("Why do cats bring dead animals?"): "Cats bring dead prey as a gift rooted in their hunting instinct, treating their owners as part of their family group to provide for.",
    _normalize("Why do cats show their belly?"): "Cats expose their belly to signal trust and relaxation, but it is not always an invitation to pet — touching it may trigger a defensive response.",
    _normalize("Why do cats chirp at birds?"): "Cats chirp at birds as an instinctual predatory response, expressing excitement and frustration at prey they cannot reach.",
    _normalize("Why do cats sleep so much?"): "Cats sleep 12 to 16 hours a day because they are natural predators that conserve energy for short intense bursts of activity.",
    _normalize("Why do cats scratch furniture?"): "Cats scratch to shed old claw layers, stretch their muscles, and leave scent and visual territorial markings.",
    _normalize("Why do cats headbutt?"): "Cats headbutt to transfer scent from glands on their head, marking people and objects as safe and familiar.",
    _normalize("Why do cats knock things over?"): "Cats knock things over to investigate objects with their paws, satisfy hunting instincts, and get attention from their owners.",
    _normalize("Why do cats stare?"): "Cats stare to assess their environment or focus on potential prey — a prolonged unblinking stare between cats can signal a challenge.",
    _normalize("Why do cats roll over?"): "Cats roll over to show trust and playfulness, exposing their belly as a sign that they feel completely safe.",
    _normalize("Why do cats slow blink?"): "Cats slow blink at trusted humans as a sign of affection and relaxed trust — returning the slow blink signals mutual comfort.",
    _normalize("Why do cats groom themselves?"): "Cats groom to keep their coat clean, regulate body temperature, spread natural oils, and self-soothe.",
}

_FALLBACK = (
    "I can answer questions about cat behavior — for example: "
    "why cats knead, purr, rub against people, push things off tables, "
    "scratch furniture, slow blink, or bring dead animals."
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
        out = "This question is outside of my cat behavior domain."
        if len(_response_cache) < _CACHE_MAX:
            _response_cache[key] = out
        return out

    if is_food_safety(question):
        out = "I'm not able to provide food safety advice — the model powering this bot is too small to be reliably accurate on pet nutrition. For questions about what cats can or cannot eat, please consult your veterinarian."
        if len(_response_cache) < _CACHE_MAX:
            _response_cache[key] = out
        return out

    if is_greeting(question):
        out = "Meow~ I'm your cat behavior expert! Ask me why cats knead, purr, scratch, slow blink, headbutt, or do anything else curious!"
        if len(_response_cache) < _CACHE_MAX:
            _response_cache[key] = out
        return out

    # Call the model for novel in-domain questions
    answer = _call_hf(question)

    # Python backstop: re-check generated output (defense-in-depth)
    if is_safety_trigger(answer):
        answer = "I'm not able to help with that. If you're going through a difficult time, please reach out to a mental health professional or a crisis line in your area."
    elif is_out_of_scope(answer):
        answer = "This question is outside of my cat behavior domain."
    else:
        # Truncation guard: if the model ran out of tokens mid-sentence,
        # trim to the last complete sentence rather than returning a cut-off reply.
        if answer and answer[-1] not in ".?!":
            last_end = max(answer.rfind("."), answer.rfind("?"), answer.rfind("!"))
            if last_end > 0:
                answer = answer[:last_end + 1]
        # Numbered-list truncation guard: remove trailing empty list items like
        # "\n\n5." that have no content (token limit hit between number and text).
        answer = re.sub(r'\n+\d+\.\s*$', '', answer).strip()

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
    return FileResponse("index2.html")


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
