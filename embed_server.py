import base64
from typing import List, Optional, Union

import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("BAAI/bge-small-en-v1.5")
app = FastAPI()


class EmbeddingRequest(BaseModel):
    input: Union[str, List[str]]
    model: Optional[str] = None
    encoding_format: Optional[str] = "float"


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingRequest):
    texts = [req.input] if isinstance(req.input, str) else req.input
    vectors = model.encode(texts, normalize_embeddings=True)

    data = []
    for i, v in enumerate(vectors):
        if req.encoding_format == "base64":
            emb = base64.b64encode(np.asarray(v, dtype=np.float32).tobytes()).decode()
        else:
            emb = v.tolist()
        data.append({"object": "embedding", "index": i, "embedding": emb})

    return {
        "object": "list",
        "data": data,
        "model": "bge-small-en-v1.5",
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


@app.get("/v1/models")
def list_models():
    return {
        "object": "list",
        "data": [{"id": "bge-small-en-v1.5-embed", "object": "model", "created": 0, "owned_by": "local"}],
    }
