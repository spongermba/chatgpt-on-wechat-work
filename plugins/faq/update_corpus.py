import json
import asyncio
import csv
import io
from enum import Enum
from pydantic import BaseModel
from fastapi import FastAPI, File, UploadFile
from loguru import logger
from corpus_tools import corpus_tools
import helper as helper

class ErrorCodes(Enum):
    SUCCESS = 0
    PARAMS_TEXT_LEN_EXCEEDS_LIMIT = 1
    LLM_INVALID_JSON_RESULT = 2
    LLM_EXCEPTION = 3

class UpdateCorpusRequest(BaseModel):
    question: str
    answer: str

class BaseResponse(BaseModel):
    code: int
    message: str

app = FastAPI()

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/search_corpus", response_model=BaseResponse)
async def search_corpus(question: str):
    if len(question) > 150:
        return BaseResponse(code=ErrorCodes.PARAMS_TEXT_LEN_EXCEEDS_LIMIT.value, message="问题长度超过150")
    try:
        ret_value = corpus_tools.search_chroma_text(question)
        print("ret_value: {}".format(ret_value))
        if ret_value is None:
            return BaseResponse(code=ErrorCodes.LLM_INVALID_JSON_RESULT.value, message="invalid json result")
    
        return BaseResponse(code=ErrorCodes.SUCCESS.value, message=ret_value)
    except Exception as e:
        logger.error("search_corpus exception: {}".format(e))
        return BaseResponse(code=ErrorCodes.LLM_EXCEPTION.value, message="exception: {}".format(e))
    
@app.post("/update_corpus", response_model=BaseResponse)
async def update_corpus(request: UpdateCorpusRequest):
    if len(request.question) > 150 or len(request.answer) > 150:
        return BaseResponse(code=ErrorCodes.PARAMS_TEXT_LEN_EXCEEDS_LIMIT.value, message="问题或者回答长度超过150")
    try:
        corpus_tools.update_chroma_text(request.question, request.answer)
        ret_value = corpus_tools.search_chroma_text(request.question)
        print("ret_value: {}".format(ret_value))
        if ret_value != request.answer:
            return BaseResponse(code=ErrorCodes.LLM_INVALID_JSON_RESULT.value, message="invalid json result")
    
        return BaseResponse(code=ErrorCodes.SUCCESS.value, message=ret_value)
    except Exception as e:
        logger.error("update_corpus exception: {}".format(e))
        return BaseResponse(code=ErrorCodes.LLM_EXCEPTION.value, message="exception: {}".format(e))
    
@app.post("/generate_embedding_from_file", response_model=BaseResponse)
async def update_corpus_from_file(file: UploadFile):
    try:
        content = await file.read()
        data = content.decode('utf-8')
        reader = csv.reader(io.StringIO(data))
        corpus_tools.generate_embedding_from_csv_reader(reader)
        return BaseResponse(code=ErrorCodes.SUCCESS.value, message="success")
    except Exception as e:
        logger.error("update_corpus_from_file exception: {}".format(e))
        return BaseResponse(code=ErrorCodes.LLM_EXCEPTION.value, message="exception: {}".format(e))