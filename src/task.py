import os
import uuid
import logging
import requests
import contextlib
import langdetect
import urllib.request

from string import Template
from urllib.parse import urlparse
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Optional, Type, TypedDict

from .segmentors import AbstractSegmentor, get_segmentor

from .sparql_config import LANGUAGE_CODE_TO_URI, get_prefixes_for_query, GRAPHS, JOB_STATUSES, TASK_OPERATIONS
from escape_helpers import sparql_escape_uri, sparql_escape_string
from helpers import query, update


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
              BIND($uri AS ?task)
              ?task task:operation ?taskType .
            }
        """).substitute(uri=sparql_escape_uri(task_uri))
        for b in query(q, sudo=True).get('results').get('bindings'):
            candidate_cls = cls.lookup(b['taskType']['value'])
            if candidate_cls is not None:
                return candidate_cls(task_uri)
            raise RuntimeError(
                "Unknown task type {0}".format(b['taskType']['value']))
        raise RuntimeError("Task with uri {0} not found".format(task_uri))

    def change_state(self, old_state: str, new_state: str, results_container_uris: list = []) -> None:
        """Update the task status in the triplestore."""

        # Update the task status
        status_query = Template(
            get_prefixes_for_query("task", "adms") +
            """
            DELETE {
            GRAPH <""" + GRAPHS["jobs"] + """> {
                ?task adms:status ?oldStatus .
            }
            }
            INSERT {
            GRAPH <""" + GRAPHS["jobs"] + """> {
                ?task adms:status <$new_status> .
            }
            }
            WHERE {
            GRAPH <""" + GRAPHS["jobs"] + """> {
                BIND($task AS ?task)
                BIND(<$old_status> AS ?oldStatus)
                OPTIONAL { ?task adms:status ?oldStatus . }
            }
            }
            """
        )
        query_string = status_query.substitute(
            new_status=JOB_STATUSES[new_state],
            old_status=JOB_STATUSES[old_state],
            task=sparql_escape_uri(self.task_uri)
        )

        update(query_string, sudo=True)

        # Batch-insert results containers (if any)
        if results_container_uris:
            BATCH_SIZE = 50
            insert_template = Template(
                get_prefixes_for_query("task", "adms") +
                """
                INSERT {
                GRAPH <""" + GRAPHS["jobs"] + """> {
                    ?task $results_container_line .
                }
                }
                WHERE {
                    BIND($task AS ?task)
                }
                """
            )

            for i in range(0, len(results_container_uris), BATCH_SIZE):
                batch_uris = results_container_uris[i:i + BATCH_SIZE]
                results_container_line = " ;\n".join(
                    [f"task:resultsContainer {sparql_escape_uri(uri)}" for uri in batch_uris]
                )
                query_string = insert_template.substitute(
                    task=sparql_escape_uri(self.task_uri),
                    results_container_line=results_container_line
                )
                update(query_string, sudo=True)

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
        Retrieve filenames and download URLs from the task's input container,
        supporting both remote and local PDFs.

        Returns:
            Dictionary containing the lists of PDF filenames and download URLs
                Keys:
                    - "filenames": list[str]
                    - "download_urls": list[str]
        """
        q = Template(
            get_prefixes_for_query("task") +
            f"""
            SELECT ?container WHERE {{
            GRAPH <{GRAPHS["jobs"]}> {{
                $task task:inputContainer ?container .
            }}
            }}
            """
        ).substitute(task=sparql_escape_uri(self.task_uri))

        bindings = query(q, sudo=True).get("results", {}).get("bindings", [])
        if not bindings:
            return {
                "filenames": [],
                "download_urls": [],
            }

        container_uri = bindings[0]["container"]["value"]

        if self._container_has_harvest_collection(container_uri):
            return self._fetch_remote_files(container_uri)
        else:
            return self._fetch_local_files(container_uri)

    def _container_has_harvest_collection(self, container_uri: str) -> bool:
        """
        Private helper function to determine if the input container has
        a harvest collection (and thus if it consists of remote or local PDFs).

        Returns:
            Boolean indicating whether the input container
            has a harvest collection or not.
        """
        q = f"""
            {get_prefixes_for_query("task")}

            ASK {{
            GRAPH <{GRAPHS["data_containers"]}> {{
                <{container_uri}> task:hasHarvestingCollection ?c .
            }}
            }}
            """

        return query(q, sudo=True).get("boolean", False)

    def _fetch_remote_files(self, container_uri: str) -> dict[str, list[str]]:
        """
        Private helper function to retrieve
        the download urls and filenames of remote PDFs.

        Returns:
            Dictionary containing the lists of PDF filenames and download URLs
                Keys:
                    - "filenames": list[str]
                    - "download_urls": list[str]
        """
        q = f"""
            {get_prefixes_for_query("task", "dct", "nfo", "nie", "mu")}
            SELECT ?fileUrl ?uuid WHERE {{
            GRAPH <{GRAPHS["data_containers"]}> {{
                <{container_uri}> task:hasHarvestingCollection ?collection .
            }}
            GRAPH <{GRAPHS["harvest_collections"]}> {{
                ?collection dct:hasPart ?remote .
            }}
            GRAPH <{GRAPHS["remote_objects"]}> {{
                ?remote a nfo:RemoteDataObject ;
                    mu:uuid ?uuid ;
                    nie:url ?fileUrl .
            }}
            }}
            """

        bindings = query(q, sudo=True).get("results", {}).get("bindings", [])
        if not bindings:
            raise RuntimeError(
                "No remote files found in harvesting collection")

        download_urls = [b["fileUrl"]["value"] for b in bindings]
        filenames = [b["uuid"]["value"] + ".pdf" for b in bindings]

        return {
            "filenames": filenames,
            "download_urls": download_urls,
        }

    def _fetch_local_files(self, container_uri: str) -> dict[str, list[str]]:
        """
        Private helper function to retrieve
        the download urls and filenames of local PDFs.

        Returns:
            Dictionary containing the lists of PDF filenames and download URLs
                Keys:
                    - "filenames": list[str]
                    - "download_urls": list[str]
        """
        q = f"""
            {get_prefixes_for_query("task", "nfo", "nie")}
            SELECT ?fileName ?shareIri WHERE {{
            GRAPH <{GRAPHS["data_containers"]}> {{
                <{container_uri}> task:hasFile ?file .
            }}

            GRAPH <{GRAPHS["files"]}> {{
                ?file a nfo:FileDataObject ;
                    nfo:fileName ?fileName .

                ?shareIri a nfo:FileDataObject ;
                        nie:dataSource ?file .
            }}
            }}
            """

        bindings = query(q, sudo=True).get("results", {}).get("bindings", [])
        if not bindings:
            raise RuntimeError(
                "No local share:// files found in data container")

        return {
            "filenames": [b["fileName"]["value"] for b in bindings],
            "download_urls": [b["shareIri"]["value"] for b in bindings],
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
            parsed = urlparse(download_url)

            share_root = os.environ["MOUNTED_SHARE_FOLDER"]
            share_folder_name = os.path.dirname(share_root)

            if parsed.scheme == share_folder_name:
                saved_path = os.path.join(
                    share_root, parsed.netloc + parsed.path)

            else:
                extract_folder = os.path.join(share_root, "extract")
                os.makedirs(extract_folder, exist_ok=True)

                saved_path = os.path.join(
                    extract_folder, os.path.basename(filename))

                try:
                    urllib.request.urlretrieve(
                        download_url, filename=saved_path)
                except Exception as e:
                    self.logger.exception(
                        f"Exception during PDF download: {e}")

            if os.path.isfile(saved_path):
                try:
                    byte_size = os.path.getsize(saved_path)

                    with open(saved_path, "rb") as f:
                        response = requests.put(
                            os.environ["APACHE_TIKA_URL"],
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

    def create_eli_expression(self, decision: dict[str, str], language: str, manifestation_uri: str) -> str:
        """
        Function to create a single ELI expression from PDF content.

        Args:
            decision: Dictionary containing text and title of the decision
            language: String containing the language code of the extracted content
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
                    eli:title $title ;
                    eli:language $language ;
                    epvoc:expressionContent $content ;
                    eli:is_embodied_by $manif ;
                    dcterms:created "$now"^^xsd:dateTime ;
                    dcterms:modified "$now"^^xsd:dateTime .
            }}
            }}
            """
        ).substitute(
            expr=sparql_escape_uri(expression_uri),
            title=f"{sparql_escape_string(decision['title'])}@{language}",
            language=sparql_escape_uri(LANGUAGE_CODE_TO_URI.get(language)),
            content=f"{sparql_escape_string(decision['text'])}@{language}",
            manif=sparql_escape_uri(manifestation_uri),
            now=now,
        )

        update(q, sudo=True)

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

        update(q, sudo=True)

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

        update(q, sudo=True)

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
        container_uri = f"http://data.lblod.info/id/data-container/{container_id}"

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

        update(q, sudo=True)
        return container_uri

    def split_decisions(self, text: str, segmentor: AbstractSegmentor) -> list[dict[str, str]]:
        """
        Split the extracted text into individual decisions.

        Args:
            text: The full text extracted from the PDF.
            extractor: An instance of TitleExtractor to extract titles from the text.
        Returns:
            A list of dictionaries, each representing an individual decision with "text" and "title" keys.
        """

        decisions = []
        segments = segmentor.segment(text)
        print(segments)
        decision_titles = [
            segment for segment in segments if segment["label"].lower() == "title"]

        if len(decision_titles) == 0:
            return [{"text": text, "title": ""}]
        elif len(decision_titles) == 1:
            return [{"text": text, "title": decision_titles[0]["text"]}]
        else:
            decision_titles_sorted = sorted(
                decision_titles, key=lambda s: s["start"])

            intro_text = text[:decision_titles_sorted[0]["start"]]

            for i in range(len(decision_titles_sorted)):
                current_title = decision_titles_sorted[i]
                if i < len(decision_titles_sorted) - 1:
                    next_title = decision_titles_sorted[i+1]
                    decision_end = next_title["start"]
                else:
                    decision_end = len(text)

                decision_text = intro_text + \
                    text[current_title["start"]:decision_end]

                decisions.append({
                    "text": decision_text,
                    "title": current_title["text"]
                })

            return decisions

    def process(self):
        """
        Implementation of Task's process function that
         - retrieves the data from the task's input data container
         - extracts the contents from the PDFs
         - creates an ELI expression for each PDF
         - creates the task's data output container containing the expressions
        """
        input = self.fetch_data_from_input_container()

        segmentor = get_segmentor()

        extraction_results = self.extract_content_from_pdf(input)
        for extraction_result in extraction_results:
            language = langdetect.detect(extraction_result["content"])
            decisions = self.split_decisions(
                extraction_result["content"], segmentor)

            manifestation_uri = self.create_manifestation(
                extraction_result["byte_size"], extraction_result["pdf_url"])
            for decision in decisions:
                expression_uri = self.create_eli_expression(
                    decision, language, manifestation_uri)
                work_uri = self.create_eli_work(expression_uri)

                self.results_container_uris.append(
                    self.create_output_container(expression_uri))
                self.results_container_uris.append(
                    self.create_output_container(work_uri))

            self.results_container_uris.append(
                self.create_output_container(manifestation_uri))
