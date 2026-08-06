# Guaraní Morphological Adaptation Using Few-Shot Learning and RAG

This repository contains an experimental pipeline for generating morphological
adaptations in Guaraní. The system combines *few-shot* examples, grammatical
information retrieval using RAG, and a configurable language model.

## Repository Structure

| Path | Description |
| --- | --- |
| `guaranidb.py` | Splits the grammar corpus and creates a Chroma vector database. |
| `fewshot_rag_with_grammar.py` | Performs RAG retrieval, generates predictions, and computes the metrics. |
| `data/guarani.txt` | Grammar corpus used to build the vector database. |
| `data/guarani-train.tsv` | Training dataset; it must be added before running the pipeline. |
| `data/guarani-dev.tsv` | Development dataset; it must be added before running the pipeline. |
| `.env.example` | Configuration template without credentials. |
| `requirements.txt` | Python dependencies. |

The `guarani_db/` and `resultados/` directories are generated locally and are
not uploaded to GitHub.

## Requirements

- Python 3.10 or later.
- An OpenAI API key for the RAG stage.
- An API key for the provider that will generate the predictions. If OpenAI is
  selected, the same key can be used for both stages.
- The `guarani-train.tsv` and `guarani-dev.tsv` datasets.

The TSV files must be tab-separated and include the following columns:

```text
ID	Source	Change	Target
```

## Installation

1. Clone the repository and navigate to its directory:

   ```bash
   git clone https://github.com/USERNAME/REPOSITORY-NAME.git
   cd REPOSITORY-NAME
   ```

2. Create and activate a virtual environment:

   On Linux or macOS:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

   In Windows PowerShell:

   ```powershell
   py -m venv .venv
   .venv\Scripts\Activate.ps1
   ```

3. Install the dependencies:

   ```bash
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. Copy the configuration template:

   On Linux or macOS:

   ```bash
   cp .env.example .env
   ```

   In Windows PowerShell:

   ```powershell
   Copy-Item .env.example .env
   ```

5. Open `.env`, select a provider, and add the required API keys. Do not upload
   this file to GitHub.

## Provider Configuration

| `PROVIDER` | API Key Variable |
| --- | --- |
| `OPENAI` | `OPENAI_API_KEY` |
| `GEMINI` | `GOOGLE_API_KEY` |
| `GROQ` | `GROQ_API_KEY` |
| `MISTRAL` | `MISTRAL_API_KEY` |
| `DEEPSEEK` | `DEEPSEEK_API_KEY` |
| `ANTHROPIC` | `ANTHROPIC_API_KEY` |

RAG retrieval uses OpenAI regardless of the provider selected for generating
predictions. Therefore, `OPENAI_API_KEY` is required in all cases. Model names
can be changed in `.env`.

## Usage

1. Place the data files in `data/`:

   ```text
   data/
   ├── guarani.txt
   ├── guarani-dev.tsv
   └── guarani-train.tsv
   ```

2. Create the vector database:

   ```bash
   python guaranidb.py
   ```

   The script generates `guarani_db/` with the `guarani` Chroma collection.

3. Run the experiment:

   ```bash
   python fewshot_rag_with_grammar.py
   ```

The program displays BLEU, ChrF++, and accuracy scores, and saves a detailed TSV
file in `resultados/`.

## Custom Paths

Paths can be changed in `.env`:

```dotenv
DEV_PATH=data/guarani-dev.tsv
TRAIN_PATH=data/guarani-train.tsv
GUARANI_TXT_PATH=data/guarani.txt
PERSIST_DIR=guarani_db
OUTPUT_DIR=resultados
```

`DEV_PATH` and `TRAIN_PATH` also accept URLs. Relative local paths are resolved
from the script directory.

## Publishing the Repository on GitHub

After creating an empty repository on GitHub, run the following commands from
this directory:

```bash
git init
git add .
git status
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/USERNAME/REPOSITORY-NAME.git
git push -u origin main
```

Review `git status` before committing and confirm that `.env`, `guarani_db/`,
and `resultados/` do not appear in the list.


## Guarani reference corpus

The file `guarani.txt` contains selected material derived from:

Academia de la Lengua Guaraní. *Guarani Ñe’ẽtekuaa: Gramática Guarani*. 
Servilibro, Paraguay, 2023.

This file is not distributed and is not covered by the Creative Commons
or MIT licenses of this repository. All rights in the source material
remain with their respective copyright holders.

## Generated vector database

The `guarani_db/` directory is generated from the reference corpus and
is not distributed by this repository.

## AmericasNLP 2025 Shared Task 2 data

This project requires the following datasets from the AmericasNLP 2025
Shared Task 2:

- `guarani-train.tsv`
- `guarani-dev.tsv`

Users must download both files directly from their original location:

https://github.com/AmericasNLP/americasnlp2025/tree/main/ST2_EducationalMaterials/data

After downloading them, place the files in the `data/` directory of this
project.

Users are responsible for complying with any terms, licenses, attribution
requirements, or restrictions established by the dataset owners.