from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any

# Import the inference function and trigger the model loading
from app.inference import translate
from app.model_handler import MODEL # This import ensures model_handler.py runs

app = FastAPI(
    title="Transformer Translation API",
    description="An API to translate English to Hindi and visualize attention.",
    version="1.0.0"
)

# Pydantic models for request and response validation
class TranslationRequest(BaseModel):
    text: str = Field(..., min_length=1, example="My name is John.")

class TranslationResponse(BaseModel):
    translated_text: str
    source_tokens: List[str]
    target_tokens: List[str]
    attention_weights: List[List[List[float]]] # Shape: (target_len, num_heads, source_len)

@app.on_event("startup")
async def startup_event():
    if MODEL is None:
        raise RuntimeError("Model could not be loaded. Check logs for errors.")
    print("Application startup complete. Model is loaded and ready.")

@app.get("/", tags=["General"])
def read_root():
    return {"message": "Welcome to the Transformer Translation API. Go to /docs for usage."}

@app.post("/translate", response_model=TranslationResponse, tags=["Translation"])
def handle_translation(request: TranslationRequest):
    """
    Receives text, translates it, and returns the result with attention scores.
    """
    try:
        result = translate(request.text)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        # Log the exception e for debugging
        print(f"An unexpected error occurred: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")