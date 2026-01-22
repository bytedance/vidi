import argparse
import json
from collections import defaultdict

def main():
    parser = argparse.ArgumentParser(description="Evaluate Video Question Answering results.")
    parser.add_argument("--input", type=str, required=True, help="Path to the input JSON results file.")
    args = parser.parse_args()

    try:
        with open(args.input, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: File not found at {args.input}")
        return
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {args.input}")
        return

    total_correct = 0
    total_count = 0
    
    task_stats = defaultdict(lambda: {"correct": 0, "total": 0})

    for item in data:
        pred_answer = item.get("pred_answer")
        answer = item.get("answer")
        task_type = item.get("task_type", "Unknown")

        # Skip if essential data is missing
        if pred_answer is None or answer is None:
            continue

        # Normalize answers for comparison (strip whitespace, uppercase)
        pred_clean = str(pred_answer).strip().upper()
        ans_clean = str(answer).strip().upper()
        
        is_correct = pred_clean == ans_clean
        
        total_count += 1
        task_stats[task_type]["total"] += 1
        
        if is_correct:
            total_correct += 1
            task_stats[task_type]["correct"] += 1

    print("-" * 60)
    print(f"{'Task Type':<45} | {'Accuracy':<10}")
    print("-" * 60)

    # Sort task types for consistent output
    sorted_task_types = sorted(task_stats.keys())

    for task_type in sorted_task_types:
        stats = task_stats[task_type]
        accuracy = (stats["correct"] / stats["total"]) * 100 if stats["total"] > 0 else 0.0
        print(f"{task_type:<45} | {accuracy:.2f}% ({stats['correct']}/{stats['total']})")

    print("-" * 60)
    
    overall_accuracy = (total_correct / total_count) * 100 if total_count > 0 else 0.0
    print(f"{'Overall Accuracy':<45} | {overall_accuracy:.2f}% ({total_correct}/{total_count})")
    print("-" * 60)

if __name__ == "__main__":
    main()
