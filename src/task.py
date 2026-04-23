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

from decide_ai_service_base.task import DecisionTask
from decide_ai_service_base.sparql_config import LANGUAGE_CODE_TO_URI, get_prefixes_for_query, GRAPHS, TASK_OPERATIONS, AI_COMPONENTS, AGENT_TYPES, SPARQL_PREFIXES
from decide_ai_service_base.annotation import RelationExtractionAnnotation

from escape_helpers import sparql_escape_uri, sparql_escape_string
from helpers import query, update


class PdfContentExtractionTask(DecisionTask, ABC):
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
                    response = requests.get(
                        download_url,
                        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"},
                        stream=True
                    )
                    response.raise_for_status()
                    with open(saved_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            f.write(chunk)
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

    def create_eli_expression(self, decision: dict[str, str], language: str, manifestation_uri: str, expression_uuid: str, work_uri: str) -> str:
        """
        Function to create a single ELI expression from PDF content.

        Args:
            decision: Dictionary containing text and title of the decision
            language: String containing the language code of the extracted content
            manifestation_uri: URI of the manifestation that embodies this expression
            expression_uuid: Pre-generated UUID for the expression
            work_uri: URI of the ELI Work this expression realizes

        Returns:
            The created ELI expression URI
        """
        now = datetime.now(timezone.utc).astimezone(
        ).isoformat(timespec="seconds")

        expression_uri = f"{SPARQL_PREFIXES['expressions']}{expression_uuid}"

        q = Template(
            get_prefixes_for_query("mu", "eli", "epvoc", "dcterms", "xsd")
            + f"""
            INSERT DATA {{
            GRAPH <{GRAPHS["expressions"]}> {{
                $expr a eli:Expression ;
                    mu:uuid $uuid ;
                    eli:language $language ;
                    epvoc:expressionContent $content ;
                    eli:is_embodied_by $manif ;
                    eli:realizes $work ;
                    dcterms:created "$now"^^xsd:dateTime ;
                    dcterms:modified "$now"^^xsd:dateTime .
            }}
            }}
            """
        ).substitute(
            expr=sparql_escape_uri(expression_uri),
            uuid=sparql_escape_string(expression_uuid),
            language=sparql_escape_uri(LANGUAGE_CODE_TO_URI.get(language)),
            content=f"{sparql_escape_string(decision['text'])}@{language}",
            manif=sparql_escape_uri(manifestation_uri),
            work=sparql_escape_uri(work_uri),
            now=now,
        )

        update(q, sudo=True)

        return expression_uri

    def create_title_annotation(self, decision: dict[str, str], language: str, eli_expression_uri: str) -> str:
        """
        Function to create a title annotation for an ELI Expression.

        Args:
            decision: Dictionary containing text, title, title_start, and title_end of the decision
            language: String containing the language code of the extracted content
            eli_expression_uri: URI of the ELI expression for which the title annotation is created
        Returns:
            The created annotation URI
        """

        title_uri = RelationExtractionAnnotation(
            subject=eli_expression_uri,
            predicate="eli:title",
            obj=f"{sparql_escape_string(decision['title'])}@{language}",
            activity_id=self.task_uri,
            source_uri=eli_expression_uri,
            start=decision.get('title_start'),
            end=decision.get('title_end'),
            agent=AI_COMPONENTS["segmenter"],
            agent_type=AGENT_TYPES["ai_component"],
            confidence=1.0
        ).add_to_triplestore_if_not_exists()

        return title_uri

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

    def create_eli_work(self, expression_uuid: str, work_uuid: str) -> str:
        """
        Function to create a single ELI expression work.

        Args:
            expression_uuid: Pre-generated UUID of the expression that realizes this work.
            work_uuid: Pre-generated UUID for the work.

        Returns:
            The created work URI.
        """
        expression_uri = f"{SPARQL_PREFIXES['expressions']}{expression_uuid}"
        work_uri = f"{SPARQL_PREFIXES['works']}{work_uuid}"

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
                    mu:uuid $uuid ;
                    task:hasResource $resource .
            }}
            }}
            """
        ).substitute(
            container=sparql_escape_uri(container_uri),
            uuid=sparql_escape_string(container_id),
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
            A list of dictionaries, each representing an individual decision
            with "text", "title", "title_start", and "title_end" keys.
        """

        decisions = []
        segments = segmentor.segment(text)
        print(segments)
        decision_titles = [
            segment for segment in segments if segment["label"].lower() == "title"]

        if len(decision_titles) == 0:
            return [{"text": text,
                     "title": "",
                     "title_start": None,
                     "title_end": None}]

        elif len(decision_titles) == 1:
            return [{"text": text,
                     "title": decision_titles[0]["text"],
                     "title_start": decision_titles[0]["start"],
                     "title_end": decision_titles[0]["end"]}]
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
                    "title": current_title["text"],
                    "title_start": len(intro_text),
                    "title_end": len(intro_text) + (current_title["end"] - current_title["start"])
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
                expression_uuid = str(uuid.uuid4())
                work_uuid = str(uuid.uuid4())
                work_uri = self.create_eli_work(expression_uuid, work_uuid)
                expression_uri = self.create_eli_expression(
                    decision, language, manifestation_uri, expression_uuid, work_uri)                
                title_uri = self.create_title_annotation(decision,
                                                         language,
                                                         expression_uri)

                self.results_container_uris.append(
                    self.create_output_container(expression_uri))
                self.results_container_uris.append(
                    self.create_output_container(work_uri))
                self.results_container_uris.append(
                    self.create_output_container(title_uri))

            self.results_container_uris.append(
                self.create_output_container(manifestation_uri))
