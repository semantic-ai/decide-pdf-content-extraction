import os

from src.job import fail_busy_and_scheduled_tasks
from src.sparql_config import get_prefixes_for_query, GRAPHS, JOB_STATUSES, batch_size, TASK_OPERATIONS, prefixed_log 
from src.task import Task
from helpers import query

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel
import time


router = APIRouter()

class NotificationResponse(BaseModel):
    status: str
    message: str

# on startup fail existing busy tasks
fail_busy_and_scheduled_tasks()

currently_processing = None
def process_all_open_tasks():
    global currently_processing
    now = time.time
    if currently_processing:
        prefixed_log("Process already running, skipping new trigger.")
        currently_processing = now
        return
    currently_processing = now
    keep_processing_tasks_until_done()
    if currently_processing == now:
        currently_processing = None
    else:
        currently_processing = None
        process_all_open_tasks()
        
@router.post("/delta", status_code=202)
def delta(background_tasks: BackgroundTasks) -> NotificationResponse:
    # naively start processing on any incoming delta
    prefixed_log("Received delta notification with")
    background_tasks.add_task(process_all_open_tasks)

    return NotificationResponse(
        status="accepted",
        message="Processing started",
    )

def keep_processing_tasks_until_done():
    prefixed_log("Checking for open tasks...")
    has_more = True
    while has_more:
        open_tasks = find_open_tasks()
        prefixed_log(f"Found {len(open_tasks)} open tasks")
        for uri in open_tasks:
            task = Task.from_uri(uri)
            task.execute()
        has_more = len(open_tasks) > 0

def find_open_tasks():
    q = f"""
        {get_prefixes_for_query("task", "adms")}
        SELECT ?task WHERE {{
        GRAPH <{GRAPHS["jobs"]}> {{
            ?task adms:status <{JOB_STATUSES["scheduled"]}> ;
                  task:operation <{TASK_OPERATIONS["pdf_content_extraction"]}> .
        }}
        }}
        limit {batch_size}
    """
    results = query(q)
    return [
        binding["task"]["value"]
        for binding in results.get("results", {}).get("bindings", [])
    ]