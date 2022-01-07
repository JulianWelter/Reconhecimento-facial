from pydantic import BaseModel


class MyImageDTO(BaseModel):
    id: int
    img: str
    name: str
    encoded: str
