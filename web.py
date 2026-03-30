from src.job import fail_busy_and_scheduled_tasks
from decide_ai_service_base.task import Task
from src.task import PdfContentExtractionTask
from decide_ai_service_base.util import fail_busy_and_scheduled_tasks, process_open_tasks, wait_for_triplestore
from decide_ai_service_base.schema import NotificationResponse, TaskOperationsResponse
from decide_ai_service_base.task import Task

from fastapi import APIRouter, BackgroundTasks


@app.on_event("startup")
async def startup_event():
    wait_for_triplestore()
    # on startup fail existing busy tasks
    fail_busy_and_scheduled_tasks()
    # on startup also immediately start scheduled tasks
    process_open_tasks()


router = APIRouter()


@router.post("/delta", status_code=202)
async def delta(background_tasks: BackgroundTasks) -> NotificationResponse:
    background_tasks.add_task(process_open_tasks)
    return NotificationResponse(status="accepted", message="Processing started")


@router.get("/task/operations")
def get_task_operations() -> TaskOperationsResponse:
    return TaskOperationsResponse(
        task_operations=[
            clz.__task_type__ for clz in Task.supported_operations()
        ]
    )
