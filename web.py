from threading import Lock
from src.job import fail_busy_and_scheduled_tasks
from decide_ai_service_base.task import Task
from decide_ai_service_base.util import fail_busy_and_scheduled_tasks, TaskProcessor, process_open_tasks, wait_for_triplestore, write_agent_info
from decide_ai_service_base.schema import NotificationResponse, TaskOperationsResponse
from decide_ai_service_base.task import Task

from fastapi import APIRouter, BackgroundTasks
from src.task import PdfContentExtractionTask # import needed to register the task as a subclass, so the operation is found

_open_tasks_lock = Lock()

@app.on_event("startup")
async def startup_event():
    wait_for_triplestore()
    # on startup fail existing busy tasks
    fail_busy_and_scheduled_tasks()
    write_agent_info("http://lblod.data.gift/id/components/pdf-to-eli/v1.0.0")
    # on startup also immediately start scheduled tasks
    process_open_tasks(_open_tasks_lock)


router = APIRouter()


@router.post("/delta", status_code=202)
async def delta(background_tasks: BackgroundTasks) -> NotificationResponse:
    processor = TaskProcessor(_open_tasks_lock)
    background_tasks.add_task(processor)
    return NotificationResponse(status="accepted", message="Processing started")


@router.get("/task/operations")
def get_task_operations() -> TaskOperationsResponse:
    return TaskOperationsResponse(
        task_operations=[
            clz.__task_type__ for clz in Task.supported_operations()
        ]
    )
