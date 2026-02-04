import os

from src.task import Task

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
from typing import Literal


router = APIRouter()


class Value(BaseModel):
    type: str
    value: str

class Triplet(BaseModel):
    subject: Value
    predicate: Value
    object: Value
    graph: Value

class DeltaNotification(BaseModel):
    inserts: list[Triplet]
    deletes: list[Triplet]


class NotificationResponse(BaseModel):
    status: str
    message: str


@router.post("/delta", status_code=202)
async def delta(data: list[DeltaNotification], background_tasks: BackgroundTasks) -> NotificationResponse:
    print("Received delta notification with", data, "patches")
    for patch in data:
        for ins in patch.inserts:
            task = Task.from_uri(ins.subject.value)
            background_tasks.add_task(task.execute)
    return NotificationResponse(status="accepted", message="Processing started")
