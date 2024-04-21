from pydantic import BaseModel


class RequestModel(BaseModel):
    text: str


class ResponseModel(BaseModel):
    tags: list[str]
    key_words: dict[str, list[str]]
