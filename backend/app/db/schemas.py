from pydantic import BaseModel
from typing import List, Optional

class TagBase(BaseModel):
    name: str

class TagResponse(TagBase):
    id: int
    class Config:
        orm_mode = True

class TemplateBase(BaseModel):
    name: str
    latex_src: str
    preview_image: Optional[str] = None

class TemplateCreate(TemplateBase):
    tags: List[str] = []  # list of tag names

class TemplateResponse(TemplateBase):
    id: int
    tags: List[TagResponse] = []
    class Config:
        orm_mode = True
