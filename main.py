import os  # For file path manipulation
import json  # For handling JSON files
from tqdm import tqdm  # For progress display

# Custom module imports for loading data and performing retrieval
from Preprocess.data_preprocess import load_data
from Model.retrieval import bge_retrieve

# Function to test and evaluate retrieval results
def test_results(pred_path, ground_truth_path="./dataset/preliminary/ground_truths_example.json"):
    """
    Compares predicted retrieval results to ground truth answers and calculates precision.
    
    Parameters:
        pred_path (str): Path to the prediction JSON file.
        ground_truth_path (str): Path to the ground truth JSON file.

    Prints:
        Precision score as a measure of accuracy.
    """
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)
    with open(pred_path) as f:
        answer_dict = json.load(f)

    # Prepare dictionaries of ground truth and predictions for easy comparison
    ground_truth_answers = {item["qid"]: item["retrieve"] for item in ground_truth["ground_truths"]}
    prediction_answers = {item["qid"]: item["retrieve"] for item in answer_dict["answers"]}

    # Calculate precision by counting correct predictions
    correct_predictions = sum(
        1 for qid, retrieve in prediction_answers.items() if retrieve == ground_truth_answers.get(qid)
    )
    total_predictions = len(prediction_answers)

    precision = correct_predictions / total_predictions

    print(f"Precision: {precision:.4f}")

# Main function to execute retrieval and evaluation
if __name__ == "__main__":
    current_path = os.getcwd()
    question_path = os.path.join(current_path, "./dataset/preliminary/questions_example.json")
    source_path = os.path.join(current_path, "./reference")
    output_path = os.path.join(current_path, "./dataset/preliminary/pred_retrieve.json")

    # Load question data
    with open(question_path, "rb") as f:
        qs_ref = json.load(f)

    # Define paths for reference data
    source_path_insurance = os.path.join(source_path, "insurance")
    source_path_finance = os.path.join(source_path, "finance")

    # Load FAQ reference data
    with open(os.path.join(source_path, "faq/pid_map_content.json"), "rb") as f_s:
        key_to_source_dict = json.load(f_s)
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}

    # Load document text data for each category
    corpus_dict_insurance = load_data(source_path_insurance)
    corpus_dict_finance = load_data(source_path_finance)

    # Initialize the answer dictionary to store retrieval results
    answer_dict = {"answers": []}
    for q_dict in tqdm(qs_ref["questions"], total=len(qs_ref["questions"])):
        # Perform retrieval based on the category of each question
        if q_dict["category"] == "finance":
            retrieved = bge_retrieve(q_dict["query"], q_dict["source"], corpus_dict_finance, 500, 50)
            answer_dict["answers"].append({"qid": q_dict["qid"], "retrieve": retrieved})

        elif q_dict["category"] == "insurance":
            retrieved = bge_retrieve(q_dict["query"], q_dict["source"], corpus_dict_insurance, 100, 50)
            answer_dict["answers"].append({"qid": q_dict["qid"], "retrieve": retrieved})

        elif q_dict["category"] == "faq":
            corpus_dict_faq = {
                key: str(value) for key, value in key_to_source_dict.items() if key in q_dict["source"]
            }
            retrieved = bge_retrieve(q_dict["query"], q_dict["source"], corpus_dict_faq, 800, 400)
            answer_dict["answers"].append({"qid": q_dict["qid"], "retrieve": retrieved})

        else:
            raise ValueError("Something went wrong")  # Raise an error if an unexpected category is found
    
    # Save retrieval results to a JSON file
    with open(output_path, "w", encoding="utf8") as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)

    # Evaluate retrieval performance
    test_results(output_path, os.path.join(current_path, "./dataset/preliminary/ground_truths_example.json"))
