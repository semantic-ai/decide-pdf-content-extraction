from string import Template

from helpers import update, sparqlQuery, sparqlUpdate
from escape_helpers import sparql_escape_uri

from .sparql_config import get_prefixes_for_query, GRAPHS, JOB_STATUSES, TASK_OPERATIONS, prefixed_log


def fail_busy_and_scheduled_tasks():
    prefixed_log("Startup: failing busy tasks if there are any")

    q = Template(
        get_prefixes_for_query("task", "adms", "dct") +
        f"""
        DELETE {{
            GRAPH $graph {{
                ?task  adms:status ?status
            }}
        }}
        INSERT {{
            GRAPH $graph {{
            ?task adms:status {sparql_escape_uri(JOB_STATUSES["failed"])}
            }}
        }}
        WHERE  {{
            GRAPH $graph {{
                ?task a task:Task .
                ?task dct:isPartOf ?job;
                task:operation {sparql_escape_uri(TASK_OPERATIONS["pdf_content_extraction"])} ;
                adms:status ?status.
            VALUES ?status {{
                {sparql_escape_uri(JOB_STATUSES["busy"])}
            }}
            }}
        }}
        """
    ).substitute(graph=sparql_escape_uri(GRAPHS["jobs"]))

    update(q, sudo=True)