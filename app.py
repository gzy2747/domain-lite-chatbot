import re
import uuid
import uvicorn
import torch

from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM



# MODEL
MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()



# Regex Backstop
OUT_OF_SCOPE_REGEX = re.compile(
    r"(deep learning|neural networks?|machine learning|ai models?|"
    r"doctor|medical|therapy|diagnosis|treatment|"
    r"legal|lawyer|compliance|regulation|"
    r"software architecture|system design|backend)",
    re.IGNORECASE,
)


def is_out_of_scope(text: str) -> bool:
    return bool(OUT_OF_SCOPE_REGEX.search(text))



# Generator
def generate_response(question: str) -> str:

    if is_out_of_scope(question):
        return "This question is outside my analytics domain."

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
"This question is outside my analytics domain."


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
This question is outside my analytics domain.

<|user|>
{question}
<|assistant|>

""".strip()

    inputs = tokenizer(chatml_prompt, return_tensors="pt").to(device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=160,
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
    answer = answer.replace("nt|>", "").strip()
    # print("RAW OUTPUT:", repr(answer))

    if is_out_of_scope(answer):
        return "This question is outside my analytics domain."

    return answer




# FastAPI
app = FastAPI()


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
    response = generate_response(request.message)
    return ChatResponse(
        response=response,
        session_id=str(uuid.uuid4()),
    )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8080)
