import contextlib
from datetime import datetime, timezone
import logging
from abc import ABC, abstractmethod
from typing import Optional, Type
import uuid

from helpers import query
from string import Template
from escape_helpers import sparql_escape_uri, sparql_escape_string

import os
import urllib.request
import requests

from .sparql_config import get_prefixes_for_query, GRAPHS, JOB_STATUSES, TASK_OPERATIONS


class Task(ABC):
    """Base class for background tasks that process data from the triplestore."""

    def __init__(self, task_uri: str):
        super().__init__()
        self.task_uri = task_uri
        self.results_container_uri = ""
        self.logger = logging.getLogger(self.__class__.__name__)

    @classmethod
    def supported_operations(cls) -> list[Type['Task']]:
        all_ops = []
        for subclass in cls.__subclasses__():
            if hasattr(subclass, '__task_type__'):
                all_ops.append(subclass)
            else:
                all_ops.extend(subclass.supported_operations())
        return all_ops

    @classmethod
    def lookup(cls, task_type: str) -> Optional['Task']:
        """
        Yield all subclasses of the given class, per:
        """
        for subclass in cls.supported_operations():
            if hasattr(subclass, '__task_type__') and subclass.__task_type__ == task_type:
                return subclass
        return None

    @classmethod
    def from_uri(cls, task_uri: str) -> 'Task':
        """Create a Task instance from its URI in the triplestore."""
        q = Template(
            get_prefixes_for_query("adms", "task") +
            """
            SELECT ?task ?taskType WHERE {
              ?task task:operation ?taskType .
              FILTER(?task = $uri)
            }
        """).substitute(uri=sparql_escape_uri(task_uri))
        for b in query(q).get('results').get('bindings'):
            candidate_cls = cls.lookup(b['taskType']['value'])
            if candidate_cls is not None:
                return candidate_cls(task_uri)
            raise RuntimeError(
                "Unknown task type {0}".format(b['taskType']['value']))
        raise RuntimeError("Task with uri {0} not found".format(task_uri))

    def change_state(self, old_state: str, new_state: str, results_container_uri: str = "") -> None:
        """Update the task status in the triplestore."""
        query_template = Template(
            get_prefixes_for_query("task", "adms") +
            """
            DELETE {
            GRAPH <""" + GRAPHS["jobs"] + """> {
                ?task adms:status ?oldStatus .
            }
            }
            INSERT {
            GRAPH <""" + GRAPHS["jobs"] + """> {
                ?task
                $results_container_line
                adms:status <$new_status> .

            }
            }
            WHERE {
            GRAPH <""" + GRAPHS["jobs"] + """> {
                BIND($task AS ?task)
                BIND(<$old_status> AS ?oldStatus)
                OPTIONAL { ?task adms:status ?oldStatus . }
            }
            }
            """)

        results_container_line = f"task:resultsContainer {sparql_escape_uri(results_container_uri)} ;" if results_container_uri else ""

        query_string = query_template.substitute(
            new_status=JOB_STATUSES[new_state],
            old_status=JOB_STATUSES[old_state],
            task=sparql_escape_uri(self.task_uri),
            results_container_line=results_container_line)

        query(query_string)

    @contextlib.contextmanager
    def run(self):
        """Context manager for task execution with state transitions."""
        self.change_state("scheduled", "busy")
        yield
        self.change_state("busy", "success", self.results_container_uri)

    def execute(self):
        """Run the task and handle state transitions."""
        with self.run():
            self.process()

    @abstractmethod
    def process(self):
        """Process task data (implemented by subclasses)."""
        pass


class PdfContentExtractionTask(Task, ABC):
    """
    Task that processes PDFs and extracts their content to generate an ELI expression,
    stored in the task's output data container.
    """

    __task_type__ = TASK_OPERATIONS["pdf_content_extraction"]

    def __init__(self, task_uri: str):
        super().__init__(task_uri)

    def fetch_data_from_input_container(self) -> dict[str: list[str]]:
        """
        Function to retrieve the data from the task's input data container

        Returns:
            Dictionary containing the lists of PDF filenames and download URLs
        """
        q = Template(
            get_prefixes_for_query("task", "nfo") +
            f"""
            SELECT ?fileUrl, ?fileName WHERE {{
            GRAPH <{GRAPHS["jobs"]}> {{
                BIND($task AS ?task)
                ?task task:inputContainer ?container .
            }}
            GRAPH <{GRAPHS["data_containers"]}> {{
                ?container a nfo:DataContainer ;
                    task:hasFile ?file .
                ?file a nfo:FileDataObject ;
                    nfo:fileName ?fileName ;
                    nfo:fileUrl ?fileUrl .
            }}
            }}
            """
        ).substitute(task=sparql_escape_uri(self.task_uri))

        bindings = query(q).get("results", {}).get("bindings", [])
        if not bindings:
            raise RuntimeError(
                f"No input files found for task {self.task_uri}")

        return {
            "filenames": [b["fileName"]["value"] for b in bindings],
            "downloadUrls": [b["fileUrl"]["value"] for b in bindings],
        }

    def extract_content_from_pdf(self, input: dict[str: list[str]]) -> list[str]:
        """
        Function to download the PDFs and extract their content using Apache Tika

        Args:
            input: Dictionary containing the lists of PDF filenames and download URLs
                (result of fetch_data_from_input_container)

        Returns:
            List containing the PDF contents as strings
        """
        contents = []

        for i in range(len(input["filenames"])):
            filename = input["filenames"][i]
            downloadUrl = input["downloadUrls"][i]

            saved_path = "./pdfs/" + os.path.basename(filename)

            try:
                urllib.request.urlretrieve(
                    downloadUrl, filename=saved_path)
            except Exception as e:
                raise e

            if os.path.isfile(saved_path):
                try:
                    with open(saved_path, "rb") as f:
                        response = requests.put(
                            os.getenv("APACHE_TIKA_URL"),
                            data=f,
                            # Ask for plain text output
                            headers={"Accept": "text/plain"}
                        )
                        contents.append(response.content.decode("utf-8"))

                except Exception as e:
                    raise e

        return contents

    def create_eli_expressions(self, contents: list[str]) -> list[str]:
        """
        Function to create ELI expressions from the PDF contents

        Args:
            contents: List containing the PDF contents as strings
                (result of extract_content_from_pdf)

        Returns:
            List containing the ELI expression URIs
        """
        expression_uris = []

        now = datetime.now(timezone.utc).astimezone(
        ).isoformat(timespec="seconds")

        for content in contents:
            expression_uri = f"http://data.lblod.info/id/eli-expressions/{uuid.uuid4()}"
            q = Template(
                get_prefixes_for_query("eli", "epvoc", "dct", "xsd") +
                f"""
                INSERT DATA {{
                GRAPH <{GRAPHS["expressions"]}> {{
                    $expr a eli:Expression ;
                        epvoc:expressionContent $content ;
                        dct:created "$now"^^xsd:dateTime ;
                        dct:modified "$now"^^xsd:dateTime .
                }}
                }}
                """
            ).substitute(
                expr=sparql_escape_uri(expression_uri),
                content=sparql_escape_string(content),
                now=now,
            )

            query(q)
            expression_uris.append(expression_uri)

        return expression_uris

    def create_output_container(self, resources: list[str]) -> str:
        """
        Function to create an output data container
        containing the ELI expression URIs as resources

        Args:
            resources: List containing the ELI expression URIs
                (result of create_eli_expressions)

        Returns:
            String containing the URI of the output data container
        """
        container_id = str(uuid.uuid4())
        container_uri = f"http://data.lblod.info/id/data-containers/{container_id}"

        has_resource_lines = " ;\n".join(
            f"task:hasResource {sparql_escape_uri(r)}" for r in resources)

        q = Template(
            get_prefixes_for_query("task", "nfo", "mu") +
            f"""
            INSERT DATA {{
            GRAPH <{GRAPHS["data_containers"]}> {{
                $container a nfo:DataContainer ;
                    mu:uuid "$uuid" ;
                    $has_resource_lines .
            }}
            }}
            """
        ).substitute(
            container=sparql_escape_uri(container_uri),
            uuid=container_id,
            has_resource_lines=has_resource_lines
        )

        query(q)
        return container_uri

    def process(self):
        """
        Implementation of Task's process function that
         - retrieves the data from the task's input data container
         - extracts the contents from the PDFs
         - creates an ELI expression for each PDF
         - creates the task's data output containter containing the expressions
        """
        input = self.fetch_data_from_input_container()

        contents = self.extract_content_from_pdf(input)

        expression_uris = self.create_eli_expressions(contents)

        self.results_container_uri = self.create_output_container(
            expression_uris)
