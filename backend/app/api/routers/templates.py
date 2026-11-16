"""Resume templates the user can pick from."""

from fastapi import APIRouter

from app.api.deps import Db
from app.api.schemas import TemplateOut
from app.db.templates import list_templates

router = APIRouter(prefix="/templates", tags=["templates"])


@router.get("", response_model=list[TemplateOut])
def read_templates(db: Db):
    return [
        TemplateOut(id=t.id, name=t.name, tags=sorted(tag.name for tag in t.tags))
        for t in list_templates(db)
    ]
