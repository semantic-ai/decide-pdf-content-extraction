import contextlib
from datetime import datetime, timezone
import logging
from abc import ABC, abstractmethod
from typing import Optional, Type, TypedDict
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
        self.results_container_uris = []
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

    def change_state(self, old_state: str, new_state: str, results_container_uris: list = []) -> None:
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

        results_container_line = ""
        if results_container_uris:
            results_container_line = "\n".join(
                [f"task:resultsContainer {sparql_escape_uri(uri)} ;" for uri in results_container_uris])

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
        self.change_state("busy", "success", self.results_container_uris)

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
    Task that processes PDFs and extracts their content to generate an ELI manifestation,
    expression and work, stored in the task's output data container.
    """

    __task_type__ = TASK_OPERATIONS["pdf_content_extraction"]

    class PdfExtractionResult(TypedDict):
        content: str
        pdf_url: str
        byte_size: int
        filename: str

    def __init__(self, task_uri: str):
        super().__init__(task_uri)

    def fetch_data_from_input_container(self) -> dict[str, list[str]]:
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
            "download_urls": [b["fileUrl"]["value"] for b in bindings],
        }

    def extract_content_from_pdf(self, input: dict[str, list[str]]) -> list[PdfExtractionResult]:
        """
        Download the PDFs and extract their content using Apache Tika.

        Args:
            input: Dictionary containing the lists of PDF filenames and download URLs
                (result of fetch_data_from_input_container)
                Expected keys:
                    - "filenames": list[str]
                    - "download_urls": list[str]

        Returns:
            List of PdfExtractionResults containing:
                - content: extracted text
                - pdf_url: original download URL
                - byte_size: size of the downloaded PDF in bytes
                - filename: local filename used to save the PDF
        """
        results = []

        for filename, download_url in zip(input["filenames"], input["download_urls"]):
            saved_path = os.path.join("./pdfs", os.path.basename(filename))

            try:
                urllib.request.urlretrieve(download_url, filename=saved_path)
            except Exception as e:
                self.logger.exception(f"Exception during PDF download: {e}")

            if os.path.isfile(saved_path):
                try:
                    byte_size = os.path.getsize(saved_path)

                    with open(saved_path, "rb") as f:
                        response = requests.put(
                            os.getenv("APACHE_TIKA_URL"),
                            data=f,
                            headers={"Accept": "text/plain"},
                        )
                        response.raise_for_status()
                        content = response.content.decode("utf-8")

                    results.append(
                        {
                            "content": content,
                            "pdf_url": download_url,
                            "byte_size": byte_size,
                            "filename": saved_path,
                        }
                    )
                except Exception as e:
                    self.logger.exception(
                        f"Exception during retrieving content of PDF: {e}")

        return results

    def create_eli_expression(self, content: str, manifestation_uri: str) -> str:
        """
        Function to create a single ELI expression from PDF content.

        Args:
            content: Text extracted from the PDF
            manifestation_uri: URI of the manifestation that embodies this expression

        Returns:
            The created ELI expression URI
        """
        now = datetime.now(timezone.utc).astimezone(
        ).isoformat(timespec="seconds")

        expression_uri = f"http://data.lblod.info/id/expressions/{uuid.uuid4()}"

        q = Template(
            get_prefixes_for_query("eli", "epvoc", "dcterms", "xsd")
            + f"""
            INSERT DATA {{
            GRAPH <{GRAPHS["expressions"]}> {{
                $expr a eli:Expression ;
                    epvoc:expressionContent $content ;
                    eli:is_embodied_by $manif ;
                    dcterms:created "$now"^^xsd:dateTime ;
                    dcterms:modified "$now"^^xsd:dateTime .
            }}
            }}
            """
        ).substitute(
            expr=sparql_escape_uri(expression_uri),
            content=sparql_escape_string(content),
            manif=sparql_escape_uri(manifestation_uri),
            now=now,
        )

        query(q)

        return expression_uri

    def create_manifestation(self, byte_size: int, pdf_url: str) -> str:
        """
        Function to create a single ELI manifestation.

        Args:
            byte_size: Size of the file in bytes.
            pdf_url: URL to the PDF.

        Returns:
            The created manifestation URI.
        """
        now = datetime.now(timezone.utc).astimezone(
        ).isoformat(timespec="seconds")

        manifestation_uuid = str(uuid.uuid4())
        manifestation_uri = f"http://data.lblod.info/id/manifestations/{manifestation_uuid}"

        q = Template(
            get_prefixes_for_query("eli", "epvoc", "dcterms", "xsd", "mu")
            + f"""
            INSERT DATA {{
            GRAPH <{GRAPHS["manifestations"]}> {{
                $manif a eli:Manifestation ;
                    mu:uuid $uuid ;
                    dcterms:created "$now"^^xsd:dateTime ;
                    dcterms:modified "$now"^^xsd:dateTime ;
                    eli:media_type "application/pdf" ;
                    epvoc:byteSize $byte_size ;
                    eli:is_exemplified_by $pdf_url .

            }}
            }}
            """
        ).substitute(
            manif=sparql_escape_uri(manifestation_uri),
            uuid=sparql_escape_string(manifestation_uuid),
            byte_size=str(byte_size),
            pdf_url=sparql_escape_uri(pdf_url),
            now=now,
        )

        query(q)

        return manifestation_uri

    def create_eli_work(self, expression_uri: str) -> str:
        """
        Function to create a single ELI expression work.

        Args:
            expression_uri: URI of the expression that realizes this work.

        Returns:
            The created work URI.
        """
        work_uuid = str(uuid.uuid4())
        work_uri = f"http://data.lblod.info/id/works/{work_uuid}"

        q = Template(
            get_prefixes_for_query("eli", "mu")
            + f"""
            INSERT DATA {{
            GRAPH <{GRAPHS["works"]}> {{
                $work a eli:Work ;
                    mu:uuid $uuid ;
                    eli:is_realized_by $expr .
            }}
            }}
            """
        ).substitute(
            work=sparql_escape_uri(work_uri),
            uuid=sparql_escape_string(work_uuid),
            expr=sparql_escape_uri(expression_uri),
        )

        query(q)

        return work_uri

    def create_output_container(self, resource: str) -> str:
        """
        Function to create an output data container for an ELI manifestation, expression or work.

        Args:
            resource: String containing an ELI manifestation, expression or work URI

        Returns:
            String containing the URI of the output data container
        """
        container_id = str(uuid.uuid4())
        container_uri = f"http://data.lblod.info/id/data-containers/{container_id}"

        q = Template(
            get_prefixes_for_query("task", "nfo", "mu") +
            f"""
            INSERT DATA {{
            GRAPH <{GRAPHS["data_containers"]}> {{
                $container a nfo:DataContainer ;
                    mu:uuid "$uuid" ;
                    task:hasResource $resource .
            }}
            }}
            """
        ).substitute(
            container=sparql_escape_uri(container_uri),
            uuid=container_id,
            resource=sparql_escape_uri(resource)
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

        extraction_results = self.extract_content_from_pdf(input)

        for extraction_result in extraction_results:
            manifestation_uri = self.create_manifestation(
                extraction_result["byte_size"], extraction_result["pdf_url"])
            expression_uri = self.create_eli_expression(
                extraction_result["content"], manifestation_uri)
            work_uri = self.create_eli_work(expression_uri)

            self.results_container_uris.append(
                self.create_output_container(manifestation_uri))
            self.results_container_uris.append(
                self.create_output_container(expression_uri))
            self.results_container_uris.append(
                self.create_output_container(work_uri))
