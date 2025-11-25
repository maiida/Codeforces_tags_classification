# Technical Test: Tag Prediction

A multi-label classification system for predicting algorithmic tags on competitive programming problems from Codeforces.

## Overview

This project implements three approaches to predict tags for algorithmic problems:

1. **CodeBERT (Fine-tuned)** - A fine-tuned CodeBERT model for multi-label classification
2. **Retrieval** - Retrieval-based approach using Weaviate vector store and BGE-M3 embeddings
3. **LLM** - Few-shot prompting with Llama 3.1 8B via Groq API

### Supported Tags
- math, graphs, strings, number theory, trees, geometry, games, probabilities


## Installation

Using pyenv ensures you have the correct Python version and isolated environment:

```bash
# Clone the repository
git clone <repository-url>
cd test_technique

# Install the required Python version via pyenv
pyenv install 3.10 --skip-existing

# Make the project use Python 3.10
pyenv local 3.10

# Create a virtual environment using this Python version
python -m venv venv

# Activate the virtual environment
source venv/bin/activate  # macOS/Linux
# OR
venv\Scripts\activate     # Windows

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Setup

### 1. Environment variables

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Edit the `.env` file and add your API keys:
   - **GROQ_API_KEY**: Required for LLM model. Get your key from [Groq Console](https://console.groq.com/keys)
   - **WEAVIATE_API_KEY**: Required for retrieval model. Get from your [Weaviate Cloud Console](https://console.weaviate.cloud/)
   - **HF_TOKEN**: Optional.

3. The system automatically loads these credentials when running predictions.

### 2. Test Dataset (for evaluation)

To evaluate models, place your test CSV file in the `data/` directory. The CSV must contain the following columns:

- `description_clean` - Problem description text
- `code_clean` - Solution code
- `tags_filtered` - Ground truth tags (for evaluation metrics)

Example:
```bash
# Place your test file here
data/test_df.csv
```

Then run evaluation:
```bash
python -m src.cli evaluate --model codebert --test-file data/test_df.csv
```

**Note**: If your dataset has different column names, you'll need to rename them to match the expected format (`description_clean`, `code_clean`, `tags_filtered`).

### 3. Model Access
The fine-tuned CodeBERT model is available on HuggingFace Hub and will be automatically downloaded when you run predictions:
- **HuggingFace**: [Ahmedjr/codebert-algorithm-tagger](https://huggingface.co/Ahmedjr/codebert-algorithm-tagger)

## CLI Reference

### `predict` - Predict tags for a single problem

**Usage:**
```bash
python -m src.cli predict [OPTIONS]
```

**Options:**
- `--model {codebert,retrieval,llm}` - Model to use (default: codebert)
- `--input-file PATH` - Path to JSON file with description and code
- `--description TEXT` - Problem description text
- `--code TEXT` - Solution code 



### `evaluate` - Evaluate models on test dataset

**Usage:**
```bash
python -m src.cli evaluate [OPTIONS]
```

**Options:**
- `--model {codebert,retrieval,llm,all}` - Model(s) to evaluate (default: codebert)
- `--test-file PATH` - Path to test CSV file (required)


## Usage

### Method 1: Using JSON Input File

Create a JSON file with your problem (or use the provided `example_problem.json`):

```json
{
    "description": "problem description text here",
    "code": "solution code here (optional)"
}
```

**Run predictions:**

```bash
# CodeBERT (default model)
python -m src.cli predict --input-file example_problem.json

# Retrieval model
python -m src.cli predict --model retrieval --input-file example_problem.json

# LLM model
python -m src.cli predict --model llm --input-file example_problem.json
```

### Method 2: Direct CLI Input


**Description and code:**
```bash
python -m src.cli predict --model llm \
    --description "sort array using quicksort" \
    --code "def quicksort(arr): pass"
```



### Evaluate on Test Dataset

```bash
# Evaluate single model
python -m src.cli evaluate --model codebert --test-file data/test_df.csv

# Evaluate all models
python -m src.cli evaluate --model all --test-file data/test_df.csv
```




## Dataset


- **Training examples:** 2,147 (after filtering for focus tags)
- **Test examples:** 531 (after filtering)
- **Focus tags:** 8 algorithmic categories



## Notebooks

- `data_processing.ipynb` - Dataset loading and preprocessing
- `data_analysis.ipynb` - Exploratory data analysis and statistics
- `finetuning_notebook.ipynb` - Initial CodeBERT fine-tuning experiments
- `finetuned_Codebert.ipynb` - CodeBERT fine-tunined evaluation notebook


