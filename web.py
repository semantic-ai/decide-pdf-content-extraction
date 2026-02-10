import os

from src.sparql_config import JOB_STATUSES
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
    inserts = [
        ins
        for patch in data
        for ins in patch.inserts
    ]

    scheduled_tasks = [
        ins.subject.value
        for ins in inserts
        if ins.predicate.value == "http://www.w3.org/ns/adms#status"
        and ins.object.value == JOB_STATUSES["scheduled"]
    ]
    print("Scheduled tasks in delta:", scheduled_tasks)
    if not scheduled_tasks:
        print("delta did not contain scheduled tasks")
        return NotificationResponse(
            status="ignored",
            message="delta didn't contain download jobs, ignoring",
        )

    for uri in scheduled_tasks:
        task = Task.from_uri(uri)
        background_tasks.add_task(task.execute)

    return NotificationResponse(
        status="accepted",
        message="Processing started",
    )
