from src.job import fail_busy_and_scheduled_tasks
from src.sparql_config import get_prefixes_for_query, GRAPHS, JOB_STATUSES, TASK_OPERATIONS, prefixed_log
from src.task import Task
from helpers import query, log
import time

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

def wait_for_triplestore():
    triplestore_live = False
    log("Waiting for triplestore...")
    while not triplestore_live:
        try:
            result = query(
                """
                SELECT ?s WHERE {
                ?s ?p ?o.
                } LIMIT 1""",
            )
            if result["results"]["bindings"][0]["s"]["value"]:
                triplestore_live = True
            else:
                raise Exception("triplestore not ready yet...")
        except Exception as _e:
            log("Triplestore not live yet, retrying...")
            time.sleep(1)
    log("Triplestore ready!")


@app.on_event("startup")
async def startup_event():
    wait_for_triplestore()
    # on startup fail existing busy tasks
    fail_busy_and_scheduled_tasks()
    # on startup also immediately start scheduled tasks
    process_open_tasks()


router = APIRouter()


class NotificationResponse(BaseModel):
    status: str
    message: str


@router.post("/delta", status_code=202)
def delta(background_tasks: BackgroundTasks) -> NotificationResponse:
    # naively start processing on any incoming delta
    prefixed_log("Received delta notification with")
    background_tasks.add_task(process_open_tasks)
    return NotificationResponse(
        status="accepted",
        message="Processing started",
    )

@router.get("/test", status_code=202)
def testendpoint():
    res = query("SELECT * WHERE { ?s ?p ?o} LIMIT 1", sudo=True)
    return {
        "success": True,
        "res": res
    }

def process_open_tasks():
    prefixed_log("Checking for open tasks...")
    uri = get_one_open_task()
    while uri is not None:
        prefixed_log("Processing {uri}")
        task = Task.from_uri(uri)
        task.execute()
        uri = get_one_open_task()


def get_one_open_task() -> str | None:
    q = f"""
        {get_prefixes_for_query("task", "adms")}
        SELECT ?task WHERE {{
        GRAPH <{GRAPHS["jobs"]}> {{
            ?task adms:status <{JOB_STATUSES["scheduled"]}> ;
                  task:operation <{TASK_OPERATIONS["pdf_content_extraction"]}> .
        }}
        }}
        limit 1
    """
    results = query(q, sudo=True)
    bindings = results.get("results", {}).get("bindings", [])
    return bindings[0]["task"]["value"] if bindings else None
