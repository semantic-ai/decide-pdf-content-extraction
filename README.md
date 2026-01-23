# decide-pdf-content-extraction
This service allows to extract the content from remote or local PDfs. For remote PDFs, there is an extra step in which they are first downloaded before they are processed. Furthermore, the service can process multiple PDFs in one go.

## Set-up
1. Clone the repository [lblod/app-decide](https://github.com/lblod/app-decide), expose the containers so that they can communicate with each other, and then run both containers. 

   Exposing the containers was done by adding a file 'docker-compose.override.yaml' to the lblod/app-decide repo containing:
   ```
   services:
     virtuoso:
       networks:
         - decide
       ports:
         - "8890:8890"
   
   networks:
     decide:
       external: true
   ```
   Create the 'decide' Docker network using the following command:
   ```
   docker network create decide
   ```

3. Mount the folder data/files in the lblod/app-decide repo as a volume and add the mounted path as the environment variable 'MOUNTED_SHARE_FOLDER'. This is the location where the local PDFs must be stored, whereas the remote PDFs will be saved in the folder 'extract' at that location.
   
## Running
Run the container using 
```
docker compose up -d # run without -d flag when you don't want to run it in the background
```

### Example 1 - remote file
Open your local SPARQL query editor (by default configured to run on http://localhost:8890/sparql as set by lblod/app-decide), and run the following query to create a Task to extract the content from a remote PDF:
```
PREFIX adms: <http://www.w3.org/ns/adms#>
PREFIX task: <http://lblod.data.gift/vocabularies/tasks/>
PREFIX dct:  <http://purl.org/dc/terms/>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
PREFIX nfo:  <http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#>
PREFIX nie:  <http://www.semanticdesktop.org/ontologies/2007/01/19/nie#>
PREFIX mu:   <http://mu.semte.ch/vocabularies/core/>

INSERT DATA {

  GRAPH <http://mu.semte.ch/graphs/jobs> {
    <http://data.lblod.info/id/tasks/demo-pdf-remote>
      a task:Task ;
      mu:uuid "demo-pdf-remote" ;
      adms:status <http://redpencil.data.gift/id/concept/JobStatus/scheduled> ;
      task:operation <http://lblod.data.gift/id/jobs/concept/TaskOperation/taskop:pdf-to-eli> ;
      task:inputContainer <http://data.lblod.info/id/data-container/demo-remote> ;
      dct:created "2025-10-31T09:00:00Z"^^xsd:dateTime .
  }

  GRAPH <http://mu.semte.ch/graphs/data-containers> {
    <http://data.lblod.info/id/data-container/demo-remote>
      a nfo:DataContainer ;
      mu:uuid "demo-remote" ;
      task:hasHarvestingCollection
        <http://lblod.data.gift/id/harvest-collections/demo-collection> .
  }

  GRAPH <http://mu.semte.ch/graphs/harvest-collections> {
    <http://lblod.data.gift/id/harvest-collections/demo-collection>
      a <http://lblod.data.gift/vocabularies/harvesting/HarvestingCollection> ;
      mu:uuid "demo-collection" ;
      dct:hasPart <http://lblod.data.gift/id/remote-data-objects/demo-pdf-1> ,
                  <http://lblod.data.gift/id/remote-data-objects/demo-pdf-2> .
  }

  GRAPH <http://mu.semte.ch/graphs/remote-objects> {
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
          "graph": { "type": "uri", "value": "http://mu.semte.ch/graphs/jobs" }
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
PREFIX task: <http://lblod.data.gift/vocabularies/tasks/>
PREFIX dct:  <http://purl.org/dc/terms/>
PREFIX xsd:  <http://www.w3.org/2001/XMLSchema#>
PREFIX nfo:  <http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#>
PREFIX nie:  <http://www.semanticdesktop.org/ontologies/2007/01/19/nie#>
PREFIX mu:   <http://mu.semte.ch/vocabularies/core/>

INSERT DATA {

  GRAPH <http://mu.semte.ch/graphs/jobs> {
    <http://data.lblod.info/id/tasks/demo-pdf-local>
      a task:Task ;
      mu:uuid "demo-pdf-local" ;
      adms:status <http://redpencil.data.gift/id/concept/JobStatus/scheduled> ;
      task:operation <http://lblod.data.gift/id/jobs/concept/TaskOperation/taskop:pdf-to-eli> ;
      task:inputContainer <http://data.lblod.info/id/data-container/demo-local> ;
      dct:created "2025-10-31T09:00:00Z"^^xsd:dateTime .
  }

  GRAPH <http://mu.semte.ch/graphs/data-containers> {
    <http://data.lblod.info/id/data-container/demo-local>
      a nfo:DataContainer ;
      mu:uuid "demo-local" ;
      task:hasFile <http://data.lblod.info/id/files/demo-pdf-3> .
  }

  GRAPH <http://mu.semte.ch/graphs/files> {
    <http://data.lblod.info/id/files/demo-pdf-3>
      a nfo:FileDataObject ;
      mu:uuid "demo-pdf-3" ;
      nfo:fileName "Vast_Bureau_Besluitenlijst_05-01-2022.pdf" ;
      dct:format "application/pdf" .
  }

  GRAPH <http://mu.semte.ch/graphs/files> {
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
          "graph": { "type": "uri", "value": "http://mu.semte.ch/graphs/jobs" }
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
PREFIX task: <http://lblod.data.gift/vocabularies/tasks/>

SELECT ?task ?status ?operation ?resultsContainer
WHERE {
  GRAPH <http://mu.semte.ch/graphs/jobs> {
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
PREFIX task: <http://lblod.data.gift/vocabularies/tasks/>
PREFIX dct:  <http://purl.org/dc/terms/>
PREFIX nfo:  <http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#>
PREFIX nie:  <http://www.semanticdesktop.org/ontologies/2007/01/19/nie#>

SELECT ?container ?file ?fileUrl ?fileLocation
WHERE {
  {
    GRAPH <http://mu.semte.ch/graphs/data-containers> {
      ?container a nfo:DataContainer ;
                 task:hasFile ?file .
    }

    GRAPH <http://mu.semte.ch/graphs/files> {
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
    GRAPH <http://mu.semte.ch/graphs/data-containers> {
      ?container a nfo:DataContainer ;
                 task:hasHarvestingCollection ?collection .
    }

    GRAPH <http://mu.semte.ch/graphs/harvest-collections> {
      ?collection dct:hasPart ?file .
    }

    GRAPH <http://mu.semte.ch/graphs/remote-objects> {
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
  GRAPH <http://mu.semte.ch/graphs/manifestations> {
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
  GRAPH <http://mu.semte.ch/graphs/expressions> {
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
  GRAPH <http://mu.semte.ch/graphs/works> {
    ?work a eli:Work ;
          mu:uuid ?uuid ;
          eli:is_realized_by ?expression .
  }
}
```

Check the created data output containers:
```
PREFIX task: <http://lblod.data.gift/vocabularies/tasks/>
PREFIX nfo:  <http://www.semanticdesktop.org/ontologies/2007/03/22/nfo#>
PREFIX mu:   <http://mu.semte.ch/vocabularies/core/>

SELECT ?container ?resource
WHERE {
  GRAPH <http://mu.semte.ch/graphs/data-containers> {
    ?container a nfo:DataContainer ;
               task:hasResource ?resource .
  }
}
ORDER BY ?resource
```
