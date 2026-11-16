"""Deleting LangGraph interview checkpoints (the stored interview transcripts)."""


def delete_interview_checkpoints(thread_ids: list[str]) -> None:
    if not thread_ids:
        return
    from app.core.checkpointer import get_checkpointer

    checkpointer = get_checkpointer()
    for thread_id in thread_ids:
        checkpointer.delete_thread(thread_id)
