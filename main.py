from FlagEmbedding import BGEM3FlagModel
from fastapi import FastAPI, HTTPException, Request
from starlette import status
from pydantic import BaseModel
from typing import List, Union, Optional

app = FastAPI()
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

class INPUTPARAMS(BaseModel):
    sentence = List[str]

@app.get("/encode_sentence", status_code=status.HTTP_200_OK)
async def get_sentence_embeddings(request : Request, sentence: INPUTPARAMS):
    try:
        embeddings = model.encode(sentence, batch_size=12, max_length=8192)['dense_vecs']
        return embeddings
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(e))
