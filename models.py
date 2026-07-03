from pydantic import BaseModel
from typing import List

class Plan(BaseModel):
    steps : list[str]


class Review(BaseModel):
    success : bool
    feedback : str
      