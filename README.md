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

### Indexing documents

In order to index the folder with documents, run the following command:

```
❯ python main.py index \
      --location <qdrant_storage_location> \
      --index <index_name> \
      run --path <path/to/files>                                        ─╯
```
