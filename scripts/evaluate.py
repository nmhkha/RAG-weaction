import json
import os
import sys
from pathlib import Path

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from src.retrieval.retriever import QdrantRetriever
from src.generation.llm_client import OllamaClient

def run_evaluation():
    print("Start to EVALUATION")

    # Create an internal LLM and Embedder for Ragas to use as a "Judge"
    judge_llm = ChatOpenAI(
        model="qwen2.5:1.5b",
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
    judge_embeddings = OpenAIEmbeddings(
        model="nomic-embed-text",
        base_url="http://localhost:11434/v1",
        api_key="ollama"
    )
    # Initialize the RAG system
    retriever = QdrantRetriever(collection_name="rag_challenge")
    llm_client = OllamaClient(model_name="qwen2.5:1.5b")

    # Read the exam questions
    dataset_path = Path("eval/dataset.json")
    with open(dataset_path, "r", encoding="utf-8") as f:
        eval_data = json.load(f)

    questions = []
    ground_truths = []
    answers = []
    contexts = []

    # Give the RAG system a test
    for item in eval_data:
        query = item["question"]
        print(f"Replying: {query}")
        
        # Retrieval
        results = retriever.search(query, top_k=3)
        context_list = [res["content"] for res in results]
        
        # Generation
        answer = llm_client.generate_answer(query, results)
        
        questions.append(query)
        ground_truths.append(item["ground_truth"])
        answers.append(answer)
        contexts.append(context_list)

    # Encapsulate data according to the HuggingFace Dataset standard required by Ragas
    data_dict = {
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    }
    hf_dataset = Dataset.from_dict(data_dict)

    print("In the progress")

    # Run the Ragas Evaluation
    result = evaluate(
        dataset=hf_dataset,
        metrics=[faithfulness],
        llm=judge_llm,
        embeddings=judge_embeddings
    )

    # Print the result and save
    print("\n Evaluation results:")
    print(result)
    
    output_path = Path("eval/results/ragas_score.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        # Cập nhật để lưu kết quả từ ragas (dạng dictionary)
        json.dump(result, f, indent=4)
        
    print(f"Đã lưu báo cáo chi tiết tại: {output_path}")

if __name__ == "__main__":
    run_evaluation()