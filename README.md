# VLM-RAG app

The goal of this project is to create a quick deployment of
RAG pipelines for textual and visual data.

## Features

- \[x\] Index files for chatting
- \[x\] Vector database for persistence
- \[x\] Answer user questions with a LLM
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

4. Install [ollama](ollama.ai) to run LLMs locally
   by downloading the installer from their website.
   Pull a model and test it:

```
ollama run llama3:latest
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

To find all arguments use `--help` argument of `fire`:

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

#### Ask questions (RAG)

Once documents have been indexed into a qdrant collection,
it is time to start chatting with them. This can be
accomplished by invoking the RAG pipeline through cli:

> NOTE: this step assumes `ollama` is running on the localhost.

```
❯ python main.py rag \
      --store_params '{location:localhost,index:recipe_files}' \
      run --query "how do you make a vegan lasagna?"
```

The command is supposed to output the following:

```
Based on the given context, to make a vegan lasagna, follow these steps:

1. Slice eggplants into 1/4 inch thick slices and rub both sides with salt.
2. Let the eggplant slices sit in a colander for half an hour to draw out excess wa
ter.
3. Roast the eggplant slices at 400°F (200°C) for about 20 minutes, or until they'r
e soft and lightly browned.
4. Meanwhile, make the pesto by blending together basil leaves, almond meal, nutrit
ional yeast, garlic powder, lemon juice, and salt to taste.
5. Make the macadamia nut cheese by blending cooked spinach, steamed tofu, drained
water from the tofu, macadamia nuts until smooth, and adjusting seasonings with gar
lic, lemon juice, and salt to taste.
6. Assemble the lasagna by layering roasted eggplant, pesto, and vegan macadamia nu
t cheese in a casserole dish. Top with additional cheese if desired (optional).
7. Bake at 350°F (180°C) for about 25 minutes, or until the cheese is melted and bu
bbly.
8. Serve and enjoy!
```

Continue by asking different questions.
