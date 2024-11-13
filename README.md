# finTech-Competition

This script retrieves answers from a knowledge base in finance, insurance, and FAQ categories based on provided questions. It saves the retrieval results and evaluates them by comparing to ground truth answers.

---
## Requirements

Python 3.10+
Dependencies:

```
pip install -r requirements.txt
```

Ensure the following directory structure, with `dataset` and `reference` folders in the same path as main.py

---
## Structure

1. test_results():
- Evaluates retrieval performance by comparing predicted answers with ground truth.
- Outputs precision as a metric of accuracy.

2. Main Retrieval Workflow:
- Loads question data and reference documents.
- For each question:
- - Loads appropriate reference documents based on the question category.
- - Runs retrieval with bge_retrieve.
- - Appends retrieval results to a dictionary.
- Saves retrieval results to a JSON file and evaluates precision with test_results.
---

## Usage

1. Run Retrieval and Evaluation:
- - Run the script with `python main.py`.
- - Results will be saved in ./dataset/preliminary/pred_retrieve.json.
2. Customize Retrieval Parameters:
- - Adjust chunk sizes and overlap within each bge_retrieve call based on your document structure.
