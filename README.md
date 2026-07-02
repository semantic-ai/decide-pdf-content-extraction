# decide-pdf-content-extraction
This service allows to extract the content from remote or local PDfs. For remote PDFs, there is an extra step in which they are first downloaded before they are processed. Furthermore, the service can process multiple PDFs in one go.

## Set-up
1. Add the service to your Semantic.Works application in the `docker-compose.yml`:

```
pdf-content:
    image: semanticai/pdf-content-service:0.0.12
    environment:
      SEGMENTATION__LLM__API_KEY: "SECRET"
      APACHE_TIKA_URL: "http://apache-tika:9998/tika"
      TARGET_GRAPH: http://mu.semte.ch/graphs/harvesting
      PUBLICATION_GRAPH: http://mu.semte.ch/graphs/public/pdf
      MOUNTED_SHARE_FOLDER: "/mnt/share"
    volumes:
      - ./data/files:/mnt/share
```



2. Mount the folder data/files in the lblod/app-decide repo as a volume and add the mounted path as the environment variable 'MOUNTED_SHARE_FOLDER'. This is the location where the local PDFs must be stored, whereas the remote PDFs will be saved in the folder 'extract' at that location.

3. The file [sparql_config.py](src/sparql_config.py) allows to easily configure SPARQL prefixes and URIs. In case a single graph for input and a single graph for output is desired, set the environment variables TARGET_GRAPH (input) and/or PUBLICATION_GRAPH (output).

## Configuration

The service uses a `config.json` file for segmentation settings. The `segmentation.llm` section controls which LLM segments PDF content into structured decisions. LLM calls are routed through [LangChain](https://python.langchain.com)'s `init_chat_model`, so the provider is chosen explicitly via the `provider` field (rather than inferred from the endpoint).

### Segmentation examples

#### Mistral AI (default, ships out of the box)

```json
{
  "segmentation": {
    "llm": {
      "provider": "mistralai",
      "model_name": "mistral-large-latest",
      "base_url": "https://api.mistral.ai/v1",
      "temperature": 0.1
    },
    "max_new_tokens": 250000,
    "text_limit_chars": 1000000
  }
}
```

- `provider`: `mistralai` (uses the bundled `langchain-mistralai` package)
- `base_url`: `https://api.mistral.ai/v1`
- API key: provide via the `SEGMENTATION__LLM__API_KEY` environment variable (preferred), or an `api_key` field under `llm`

#### Gemma (local HuggingFace model, ships out of the box)

For fully local segmentation using a fine-tuned Gemma model (no external API), set `model_name` to the Gemma model — no `provider`/`base_url`/`api_key` needed:

```json
{
  "segmentation": {
    "llm": {
      "model_name": "wdmuer/decide-marked-segmentation",
      "temperature": 0.1
    },
    "max_new_tokens": 4096,
    "text_limit_chars": 16000
  }
}
```

The model is loaded via the HuggingFace `transformers` library.

#### Azure OpenAI (requires an extra package)

> **⚠️ Not bundled:** Azure OpenAI needs the `langchain-openai` integration package, which is **not** in this service's `requirements.txt`. You must add `langchain-openai` to `requirements.txt` and rebuild the image before this provider will work.

```json
{
  "segmentation": {
    "llm": {
      "provider": "azure_openai",
      "model_name": "your-azure-deployment-name",
      "base_url": "https://YOUR_RESOURCE.openai.azure.com/",
      "temperature": 0.0
    },
    "max_new_tokens": 128000,
    "text_limit_chars": 100000
  }
}
```

- `provider`: `azure_openai`
- `model_name`: your Azure **deployment** name
- `base_url`: your Azure OpenAI resource URL (must contain `azure.com`)
- API key: supplied via `SEGMENTATION__LLM__API_KEY`
- Depending on your Azure setup you may also need to supply an API version to `init_chat_model` (e.g. via the `OPENAI_API_VERSION` environment variable)

> **Other providers (Ollama, OpenAI, …):** LangChain supports these too, but only `langchain-mistralai` is bundled in this service's `requirements.txt`. To use another provider, add its integration package (e.g. `langchain-ollama`, `langchain-openai`) to `requirements.txt`, then set `provider` accordingly (e.g. `"ollama"`, `"openai"`) with the matching `model_name`/`base_url`.

### Configuration reference

`segmentation.llm.*`:

| Field | Type | Default | Description |
|---|---|---|---|
| `provider` | string | `"mistralai"` | LangChain provider name (`mistralai`, `ollama`, `openai`, …) |
| `model_name` | string | `"mistral-large-latest"` | Model name for the provider; set to a HuggingFace model (e.g. `wdmuer/decide-marked-segmentation`) to use the local Gemma path |
| `api_key` | string \| null | `null` | API key (preferably supplied via `SEGMENTATION__LLM__API_KEY`) |
| `base_url` | string \| null | `null` | Base URL of the LLM endpoint |
| `temperature` | float | `0.0` | Generation temperature |
| `max_retries` | int | `3` | Retry attempts on LLM failure |
| `retry_delay` | float | `15.0` | Seconds between retries |

`segmentation.*`:

| Field | Type | Default | Description |
|---|---|---|---|
| `max_new_tokens` | int | `25000` | Max output tokens (must be >= `text_limit_chars / 4`) |
| `text_limit_chars` | int | `100000` | Max input characters sent to the LLM (longer documents are truncated) |

### Environment Variables

Config values can be overridden via environment variables using the `SEGMENTATION__` prefix, with `__` (double underscore) for nesting:

```
SEGMENTATION__LLM__API_KEY="your-api-key"
SEGMENTATION__LLM__PROVIDER="mistralai"
SEGMENTATION__LLM__MODEL_NAME="mistral-large-latest"
SEGMENTATION__LLM__BASE_URL="https://api.mistral.ai/v1"
```

## Running
Run the container using 
```
docker compose up -d # run without -d flag when you don't want to run it in the background
```

### Example 1 - remote file
Open your local SPARQL query editor (by default configured to run on http://localhost:8890/sparql as set by lblod/app-decide), and run the following query to create a Task to extract the content from a remote PDF:
```
PREFIX adms: <http://www.w3.org/ns/adms#>
PREFIX task: <http://redpencil.data.gift/vocabularies/tasks/>
PREFIX dct:  <http://purl.org/dc/terms/>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
PREFIX nfo:  <http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#>
PREFIX nie:  <http://www.semanticdesktop.org/ontologies/2007/01/19/nie#>
PREFIX mu:   <http://mu.semte.ch/vocabularies/core/>

INSERT DATA {

  GRAPH <http://mu.semte.ch/graphs/harvesting> {
    <http://data.lblod.info/id/tasks/demo-pdf-remote>
      a task:Task ;
      mu:uuid "demo-pdf-remote" ;
      adms:status <http://redpencil.data.gift/id/concept/JobStatus/scheduled> ;
      task:operation <http://lblod.data.gift/id/jobs/concept/TaskOperation/taskop:pdf-to-eli> ;
      task:inputContainer <http://data.lblod.info/id/data-container/demo-remote> ;
      dct:created "2025-10-31T09:00:00Z"^^xsd:dateTime .
  }

  GRAPH <http://mu.semte.ch/graphs/harvesting> {
    <http://data.lblod.info/id/data-container/demo-remote>
      a nfo:DataContainer ;
      mu:uuid "demo-remote" ;
      task:hasHarvestingCollection
        <http://lblod.data.gift/id/harvest-collections/demo-collection> .
  }

  GRAPH <http://mu.semte.ch/graphs/harvesting> {
    <http://lblod.data.gift/id/harvest-collections/demo-collection>
      a <http://lblod.data.gift/vocabularies/harvesting/HarvestingCollection> ;
      mu:uuid "demo-collection" ;
      dct:hasPart <http://lblod.data.gift/id/remote-data-objects/demo-pdf-1> ,
                  <http://lblod.data.gift/id/remote-data-objects/demo-pdf-2> .
  }

  GRAPH <http://mu.semte.ch/graphs/harvesting> {
    <http://lblod.data.gift/id/remote-data-objects/demo-pdf-1>
      a nfo:RemoteDataObject ;
      mu:uuid "demo-pdf-1" ;
      nie:url <https://lblod.ronse.be/LBLODWeb/Home/Overzicht/a1d87454f6a04638716e20a4336bb5783a2127af73425591f1b6fb48ac1b14cf/GetPublication/?filename=Agenda_Gemeenteraad_20-11-2023_Agenda.pdf> .

    <http://lblod.data.gift/id/remote-data-objects/demo-pdf-2>
      a nfo:RemoteDataObject ;
      mu:uuid "demo-pdf-2" ;
      nie:url <https://lblod.zele.be/LBLODWeb/Home/Overzicht/8588d296ef5c88d56105c79706a1ff64aa495d415a1b05a593e91e768e94dba8/GetPublication/?filename=BesluitenLijstUitspraak_Vast%20bureau_03-01-2022_Besluitenlijst.pdf> .
  }
}
```
**Note that the query contains multiple PDF.**

Trigger this task using
```
curl -X POST http://localhost:8080/delta \
  -H "Content-Type: application/json" \
  -d '[
    {
      "inserts": [
        {
          "subject": { "type": "uri", "value": "http://data.lblod.info/id/tasks/demo-pdf-remote" },
          "predicate": { "type": "uri", "value": "http://www.w3.org/ns/adms#status" },
          "object": { "type": "uri", "value": "http://redpencil.data.gift/id/concept/JobStatus/scheduled" },
          "graph": { "type": "uri", "value": "http://mu.semte.ch/graphs/harvesting" }
        }
      ],
      "deletes": []
    }
  ]'
```
The PDFs will be stored in the 'MOUNTED_SHARE_FOLDER', and the corresponding ELI manifestations, expressions and works will be stored in the triple store hosted by lblod/app-decide container.

### Example 2 - local file
Open your local SPARQL query editor (by default configured to run on http://localhost:8890/sparql as set by lblod/app-decide), and run the following query to create a Task to extract the content from a local PDF:

```
PREFIX adms: <http://www.w3.org/ns/adms#>
PREFIX task: <http://redpencil.data.gift/vocabularies/tasks/>
PREFIX dct:  <http://purl.org/dc/terms/>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
PREFIX nfo:  <http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#>
PREFIX nie:  <http://www.semanticdesktop.org/ontologies/2007/01/19/nie#>
PREFIX mu:   <http://mu.semte.ch/vocabularies/core/>

INSERT DATA {

  GRAPH <http://mu.semte.ch/graphs/harvesting> {
    <http://data.lblod.info/id/tasks/demo-pdf-local>
      a task:Task ;
      mu:uuid "demo-pdf-local" ;
      adms:status <http://redpencil.data.gift/id/concept/JobStatus/scheduled> ;
      task:operation <http://lblod.data.gift/id/jobs/concept/TaskOperation/taskop:pdf-to-eli> ;
      task:inputContainer <http://data.lblod.info/id/data-container/demo-local> ;
      dct:created "2025-10-31T09:00:00Z"^^xsd:dateTime .
  }

  GRAPH <http://mu.semte.ch/graphs/harvesting> {
    <http://data.lblod.info/id/data-container/demo-local>
      a nfo:DataContainer ;
      mu:uuid "demo-local" ;
      task:hasFile <http://data.lblod.info/id/files/demo-pdf-3> .
  }

  GRAPH <http://mu.semte.ch/graphs/harvesting> {
    <http://data.lblod.info/id/files/demo-pdf-3>
      a nfo:FileDataObject ;
      mu:uuid "demo-pdf-3" ;
      nfo:fileName "Vast_Bureau_Besluitenlijst_05-01-2022.pdf" ;
      dct:format "application/pdf" .
  }

  GRAPH <http://mu.semte.ch/graphs/harvesting> {
    <share://extract/Vast_Bureau_Besluitenlijst_05-01-2022.pdf>
      a nfo:FileDataObject ;
      nie:dataSource <http://data.lblod.info/id/files/demo-pdf-3> ;
      nfo:fileName "Vast_Bureau_Besluitenlijst_05-01-2022.pdf" .
  }
}
```
**Make sure that the mentioned file is present in the 'MOUNTED_SHARE_FOLDER'.**

Trigger this task using
```
curl -X POST http://localhost:8080/delta \
  -H "Content-Type: application/json" \
  -d '[
    {
      "inserts": [
        {
          "subject": { "type": "uri", "value": "http://data.lblod.info/id/tasks/demo-pdf-local" },
          "predicate": { "type": "uri", "value": "http://www.w3.org/ns/adms#status" },
          "object": { "type": "uri", "value": "http://redpencil.data.gift/id/concept/JobStatus/scheduled" },
          "graph": { "type": "uri", "value": "http://mu.semte.ch/graphs/harvesting" }
        }
      ],
      "deletes": []
    }
  ]'
```
The corresponding ELI manifestations, expressions and works will be stored in the triple store hosted by lblod/app-decide container.

### SPARQL queries to verify the process:
Check the tasks after inserting them (including data output containers):
```
PREFIX adms: <http://www.w3.org/ns/adms#>
PREFIX task: <http://redpencil.data.gift/vocabularies/tasks/>

SELECT ?task ?status ?operation ?resultsContainer
WHERE {
  GRAPH <http://mu.semte.ch/graphs/harvesting> {
    ?task a task:Task ;
          adms:status ?status ;
          task:operation ?operation .

    OPTIONAL { ?task task:resultsContainer ?resultsContainer . }
  }
}
ORDER BY ?task
```

Check the files to be processed:
```
PREFIX task: <http://redpencil.data.gift/vocabularies/tasks/>
PREFIX dct:  <http://purl.org/dc/terms/>
PREFIX nfo:  <http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#>
PREFIX nie:  <http://www.semanticdesktop.org/ontologies/2007/01/19/nie#>

SELECT ?container ?file ?fileUrl ?fileLocation
WHERE {
  {
    GRAPH <http://mu.semte.ch/graphs/harvesting> {
      ?container a nfo:DataContainer ;
                 task:hasFile ?file .
    }

    GRAPH <http://mu.semte.ch/graphs/harvesting> {
      ?file a nfo:FileDataObject .
      OPTIONAL {
        ?fileLocation a nfo:FileDataObject ;
                      nie:dataSource ?file .
      }
    }
  }
  UNION
  {
    # --- Remote file case (your second query) ---
    GRAPH <http://mu.semte.ch/graphs/harvesting> {
      ?container a nfo:DataContainer ;
                 task:hasHarvestingCollection ?collection .
    }

    GRAPH <http://mu.semte.ch/graphs/harvesting> {
      ?collection dct:hasPart ?file .
    }

    GRAPH <http://mu.semte.ch/graphs/harvesting> {
      ?file a nfo:RemoteDataObject ;
            nie:url ?fileUrl .
    }
  }
}
ORDER BY ?fileLocation
```

Check the created ELI manifestations:
```
PREFIX eli:   <http://data.europa.eu/eli/ontology#>
PREFIX epvoc:<https://data.europarl.europa.eu/def/epvoc#>
PREFIX dct:   <http://purl.org/dc/terms/>
PREFIX mu:    <http://mu.semte.ch/vocabularies/core/>

SELECT ?manifestation ?uuid ?mediaType ?byteSize ?pdfUrl ?created ?modified
WHERE {
  GRAPH <http://mu.semte.ch/graphs/public/pdf> {
    ?manifestation a eli:Manifestation ;
                   mu:uuid ?uuid ;
                   eli:media_type ?mediaType ;
                   epvoc:byteSize ?byteSize ;
                   eli:is_exemplified_by ?pdfUrl ;
                   dct:created ?created ;
                   dct:modified ?modified .
  }
}
ORDER BY DESC(?created)
```

Check the created ELI expressions:
```
PREFIX eli:   <http://data.europa.eu/eli/ontology#>
PREFIX epvoc:<https://data.europarl.europa.eu/def/epvoc#>
PREFIX dct:   <http://purl.org/dc/terms/>

SELECT ?expr ?content ?created ?modified
WHERE {
  GRAPH <http://mu.semte.ch/graphs/public/pdf> {
    ?expr a eli:Expression ;
          epvoc:expressionContent ?content ;
          dct:created ?created ;
          dct:modified ?modified .
  }
}
ORDER BY DESC(?created)
```

Check the created ELI works:
```
PREFIX eli: <http://data.europa.eu/eli/ontology#>
PREFIX mu:  <http://mu.semte.ch/vocabularies/core/>

SELECT ?work ?uuid ?expression
WHERE {
  GRAPH <http://mu.semte.ch/graphs/public/pdf> {
    ?work a eli:Work ;
          mu:uuid ?uuid ;
          eli:is_realized_by ?expression .
  }
}
```

Check the created data output containers:
```
PREFIX task: <http://redpencil.data.gift/vocabularies/tasks/>
PREFIX nfo:  <http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#>
PREFIX mu:   <http://mu.semte.ch/vocabularies/core/>

SELECT ?container ?resource
WHERE {
  GRAPH <http://mu.semte.ch/graphs/harvesting> {
    ?container a nfo:DataContainer ;
               task:hasResource ?resource .
  }
}
ORDER BY ?resource
```
