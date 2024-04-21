from pydantic import BaseModel


class RequestModel(BaseModel):
    text: str


class ResponseModel(BaseModel):
    tags: list[str]
