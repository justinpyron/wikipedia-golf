from pydantic import BaseModel
from pydantic_ai.messages import ModelMessage


class WikiGolfEvalInput(BaseModel):
    origin: str
    destination: str


class WikiGolfEvalOutput(BaseModel):
    path: list[str]
    messages: list[ModelMessage]
