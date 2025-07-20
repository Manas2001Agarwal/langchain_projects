from pydantic import BaseModel, Field
from typing import Literal

class EvaluateContextPrecision(BaseModel):
    binary_context_precision:Literal["yes","no"] = Field(description="Documents are relevant to the question, 'yes' or 'no'")
    cosine_similarity: float = Field(description = "similarity score between context and query based on cosine similarity metric")
    bm25_score: float = Field(description = "keyword search score based on bm25 algorithm")
    explanation: str = Field(description="explanation for scores and yes/no verdict given by model")