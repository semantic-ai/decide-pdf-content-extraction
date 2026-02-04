"""
SPARQL Configuration and Constants

This module centralizes all SPARQL prefixes, URIs, and constants used throughout
the codebase. By maintaining these in one place, updates to URIs or prefixes only
need to be made once, reducing maintenance burden and preventing inconsistencies.
"""

import os

# ==============================================================================
# SPARQL NAMESPACE PREFIXES
# ==============================================================================
# Maps prefix names to their full URIs for use in SPARQL queries

SPARQL_PREFIXES = {
    "mu": "http://mu.semte.ch/vocabularies/core/",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "airo": "https://w3id.org/airo#",
    "example": "http://www.example.org/",
    "ex": "http://example.org/",
    "prov": "http://www.w3.org/ns/prov#",
    "lblod": "https://data.vlaanderen.be/ns/lblod#",
    "oa": "http://www.w3.org/ns/oa#",
    "dct": "http://purl.org/dc/terms/",
    "dcterms": "http://purl.org/dc/terms/",
    "skolem": "http://www.example.org/id/.well-known/genid/",
    "nif": "http://persistence.uni-leipzig.org/nlp2rdf/ontologies/nif-core#",
    "locn": "http://www.w3.org/ns/locn#",
    "geosparql": "http://www.opengis.net/ont/geosparql#",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "skos": "http://www.w3.org/2004/02/skos/core#",
    "adms": "http://www.w3.org/ns/adms#",
    "task": "http://redpencil.data.gift/vocabularies/tasks/",
    "nfo": "http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#",
    "eli": "http://data.europa.eu/eli/ontology#",
    "ns1": "http://www.w3.org/ns/dqv#",
    "ns2": "https://w3id.org/okn/o/sd#",
    "ns3": "https://w3id.org/airo#",
    "schema": "https://schema.org/",
    "epvoc": "https://data.europarl.europa.eu/def/epvoc#",
    "nie": "http://www.semanticdesktop.org/ontologies/2007/01/19/nie#",
    "harvesting": "http://lblod.data.gift/vocabularies/harvesting/"
}

# ==============================================================================
# GRAPH URIs
# ==============================================================================
# Named graphs in the RDF store

TARGET_GRAPH = os.getenv("TARGET_GRAPH", None)
PUBLICATION_GRAPH = os.getenv("PUBLICATION_GRAPH", None)

GRAPHS = {
    # INPUT GRAPHS
    "jobs": TARGET_GRAPH if TARGET_GRAPH else "http://mu.semte.ch/graphs/jobs",
    "data_containers": TARGET_GRAPH if TARGET_GRAPH else "http://mu.semte.ch/graphs/data-containers",
    "harvest_collections": TARGET_GRAPH if TARGET_GRAPH else "http://mu.semte.ch/graphs/harvest-collections",
    "remote_objects": TARGET_GRAPH if TARGET_GRAPH else "http://mu.semte.ch/graphs/remote-objects",
    "files": TARGET_GRAPH if TARGET_GRAPH else "http://mu.semte.ch/graphs/files",
    # OUTPUT GRAPHS
    "expressions": PUBLICATION_GRAPH if PUBLICATION_GRAPH else "http://mu.semte.ch/graphs/expressions",
    "works": PUBLICATION_GRAPH if PUBLICATION_GRAPH else "http://mu.semte.ch/graphs/works",
    "manifestations": PUBLICATION_GRAPH if PUBLICATION_GRAPH else "http://mu.semte.ch/graphs/manifestations",
}

# ==============================================================================
# JOB STATUS URIs
# ==============================================================================

JOB_STATUS_BASE = "http://redpencil.data.gift/id/concept/JobStatus"

JOB_STATUSES = {
    "scheduled": f"{JOB_STATUS_BASE}/scheduled",
    "busy": f"{JOB_STATUS_BASE}/busy",
    "success": f"{JOB_STATUS_BASE}/success",
    "failed": f"{JOB_STATUS_BASE}/failed",
}

# ==============================================================================
# TASK OPERATION URIs
# ==============================================================================

TASK_OPERATIONS = {
    "pdf_content_extraction": "http://lblod.data.gift/id/jobs/concept/TaskOperation/pdf-to-eli",
}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================


def get_prefixes_for_query(*prefix_names: str) -> str:
    """
    Generate a SPARQL PREFIX section for only the specified prefixes.

    Args:
        *prefix_names: Variable number of prefix names to include

    Returns:
        A string containing the requested PREFIX declarations

    Example:
        >>> query = get_prefixes_for_query("oa", "prov", "mu")
        >>> query += "SELECT ?s WHERE { ... }"
    """
    lines = []
    for prefix_name in prefix_names:
        if prefix_name in SPARQL_PREFIXES:
            uri = SPARQL_PREFIXES[prefix_name]
            lines.append("PREFIX {0}: <{1}>".format(prefix_name, uri))
    if not lines:
        raise ValueError(f"No valid prefixes found in: {prefix_names}")
    return "\n".join(lines) + "\n"
