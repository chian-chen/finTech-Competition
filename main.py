import os
import json
from tqdm import tqdm

from Preprocess.data_preprocess import load_data
from Model.retrieval import bge_retrieve


def test_results(pred_path, ground_truth_path="./dataset/preliminary/ground_truths_example.json"):
    with open(ground_truth_path) as f:
        ground_truth = json.load(f)
    with open(pred_path) as f:
        answer_dict = json.load(f)


    # Extract the relevant information from the ground truth and prediction files
    ground_truth_answers = {item["qid"]: item["retrieve"] for item in ground_truth["ground_truths"]}
    prediction_answers = {item["qid"]: item["retrieve"] for item in answer_dict["answers"]}

    # Calculate the precision
    correct_predictions = sum(
        1 for qid, retrieve in prediction_answers.items() if retrieve == ground_truth_answers.get(qid)
    )
    total_predictions = len(prediction_answers)

    precision = correct_predictions / total_predictions

    print(f"Precision: {precision:.4f}")


if __name__ == "__main__":
    current_path = os.getcwd()
    question_path = os.path.join(current_path, "./dataset/preliminary/questions_example.json")
    source_path = os.path.join(current_path, "./reference")
    output_path = os.path.join(current_path, "./dataset/preliminary/pred_retrieve.json")

    with open(question_path, "rb") as f:
        qs_ref = json.load(f)  # 讀取問題檔案

    source_path_insurance = os.path.join(source_path, "insurance")  # 設定參考資料路徑
    source_path_finance = os.path.join(source_path, "finance")  # 設定參考資料路徑

    with open(os.path.join(source_path, "faq/pid_map_content.json"), "rb") as f_s:
        key_to_source_dict = json.load(f_s)  # 讀取參考資料文件
        key_to_source_dict = {int(key): value for key, value in key_to_source_dict.items()}

    corpus_dict_insurance = load_data(source_path_insurance)
    corpus_dict_finance = load_data(source_path_finance)

    answer_dict = {"answers": []}  # 初始化字典
    for q_dict in tqdm(qs_ref["questions"], total=len(qs_ref["questions"])):
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
            raise ValueError("Something went wrong")  # 如果過程有問題，拋出錯誤
    
    with open(output_path, "w", encoding="utf8") as f:
        json.dump(answer_dict, f, ensure_ascii=False, indent=4)  # 儲存檔案，確保格式和非ASCII字符

    test_results(output_path, os.path.join(current_path, "./dataset/preliminary/ground_truths_example.json"))