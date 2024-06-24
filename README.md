# VLM-RAG app

The goal of this project is to create a quick deployment of
RAG pipelines for textual and visual data.

## Features

- \[x\] Index documents
- \[ \] Vector database
- \[ \] Chat with a LLM
- \[ \] Reference original documents in the answers
- \[ \] Guardrails to prevent inappropriate and irrelevant answers
- \[ \] Webpage
- \[ \] Docker-compose to run all services

## Usage

### Installation

1. Install [ poetry ](https://python-poetry.org/docs/basic-usage/) and use system env (in case using conda or other):

```
poetry env use system
```

2. Install torch and torchvision with conda (or other):

```
conda install pytorch torchvision -c pytorch
```

3. Install dependencies:

```
poetry install
```

### System overview

- [Haystack](https://haystack.deepset.ai/tutorials/30_file_type_preprocessing_index_pipeline)
  is used as the main backend for building indexing and RAG pipelines.
- [Qdrant](https://qdrant.tech/documentation/guides/installation/)
  is used as a vector database for storing chunk embeddings.
- [Fire](https://google.github.io/python-fire/guide/) is the main entrypoint
  to run commands from cli.
- User inputs are validated with
  [pydantic](https://docs.pydantic.dev/latest/concepts/models/).

The main pipelines are defined in the `__init__` file as following:

```
from .pipelines import RAG as rag
from .pipelines import IndexingPipeline as index
```

They can be invoked in the cli as following:

```
python main.py <pipeline_name> --pipeline-args <command> --command-args
```

### Running commands

Follow the steps below to index documents and chat with them.

#### Vector database

First, we run a qdrant vector storage with docker in order
to store document reference:

```
docker compose up -d qdrant
```

Now qdrant's api is available at the 6333 port.

#### Indexing documents

Once `qdrant` is running, we can index our documents from local filesystem.
Currently supported text formats are:

- plain text
- markdown
- pdf

In order to index a folder with documents, run the following command:

```
❯ python main.py index \
        --store_params '{location:localhost,index:recipe_files}' \
        run --path=tests/data/recipe_files

Outputs:
5 documents have been written to 'localhost'
```

As a result, we have new documents indexed to the qdrant vector db.
Later on, we will be able to interact with these documents using an LLM.
