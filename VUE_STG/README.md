# VUE-STG Benchmark
**Spatio-Temporal Grounding Evaluation**

It compares predicted tubes with ground-truth tubes and computes basic metrics across time and space.

## Features
- Load VUE-STG dataset metadata and tube annotations from CSV files  
- Frame-by-frame and volume-based tube comparison  
- Compute IoU, recall, and precision  
- Export per-query and grouped summary results as CSV files  
- Includes evaluation results for **Vidi2**, **Gemini 3 Pro**, **GPT-5**, **Qwen3-VL**

## Usage
Run evaluation with:
```bash
python evaluate.py
```

## Evaluation Results
By November 21, 2025

| Metric                | t_Precision | t_Recall | t_IoU | v_Precision | v_Recall | v_IoU | v_IoU_Int |
|-----------------------|-------------|----------|-------|-------------|----------|-------|-----------|
| **Vidi2**                 | **0.730**       | **0.598**    | **0.532** | **0.446**       | **0.363**    | **0.326** | **0.603**     |
| Gemini-3-Pro-Preview  | 0.519       | 0.353    | 0.275 | 0.090       | 0.057    | 0.046 | 0.166     |
| GPT-5                 | 0.383       | 0.195    | 0.164 | 0.130       | 0.065    | 0.055 | 0.336     |
| Qwen3-VL-32B-Instruct | 0.453       | 0.392    | 0.259 | 0.086       | 0.075    | 0.051 | 0.185     |
