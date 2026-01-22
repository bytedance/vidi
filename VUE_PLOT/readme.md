# VUE-PLOT Benchmark

VUE-PLOT consists two tracks, **Character** and **Reasoning**. **Character** focuses on real-world videos and assesses fine-grained character perception by requiring models to densely localize the speaking individual and recognize their spoken content. **Reasoning** targets high-level plot understanding and reasoning, challenging models to reason about narrative dynamics, character relationships, and editing techniques through multiple-choice visual question answering on informative video clips. 

## Evaluation

This repository provides evaluation scripts for both tracks of the VUE-PLOT benchmark.

### Prerequisites

Ensure you have the necessary Python packages installed:

```bash
pip install jiwer
```
You can obtain the raw videos either using the YouTube video IDs or, alternatively, by downloading them from the [Condensed Movies dataset](https://www.robots.ox.ac.uk/~vgg/data/condensed-movies/) homepage.

### 1. Character Understanding Track

This track evaluates the model's ability to localize speakers and recognize their speech in real-world videos.

**Script:** `character_eval.py`

**Usage:**

```bash
python character_eval.py --input_file /path/to/your/results.json
```

**Input Format:**
The input should be a JSON file where each line contains the prediction results for a video query.

**Metrics:**
- **Temporal IoU**: Average Intersection over Union of matched time segments.
- **Word Error Rate (WER)**: Word distance metrics for speech recognition.
- **Bounding Box IoU**: Spatial localization accuracy of the speaker.

The script outputs a summary of these metrics and saves detailed results to a `results` directory.

### 2. Reasoning Track (VQA)

This track evaluates high-level plot understanding using multiple-choice questions.

**Script:** `vqa_eval.py`

**Usage:**

```bash
python vqa_eval.py --input /path/to/your/results.json
```

**Input Format:**
The input should be a JSON file containing a list of objects with `pred_answer`, `answer`, and `task_type` fields.

**Metrics:**
- **Overall Accuracy**: The percentage of correctly answered questions.
- **Task-Specific Accuracy**: Accuracy broken down by task types such as:
  - Narrative and Structural Understanding
  - Perception and Understanding
  - Professional Filming and Editing Techniques
  - Social Cognition and Knowledge Integration
  - Speech, Audio, and Sound Effect Reasoning

## Evaluation Results

### 1. Character Understanding Track

| Model | Temporal IoU | Spatial IoU | WER |
|-------|--------------|---------|-----|
| Gemini-3-Pro-Preview | 0.6833 | 0.1324 | 0.2900 |
| Qwen3-Omni | 0.5068 | 0.0680 | 0.5812 |
| Vidi2.5 | 0.7163 | 0.5589 | 0.2320 |

### 2. Reasoning Track (VQA)

| Model | Overall | Narrative & Structural | Perception & Understanding | Filming & Editing | Social Cognition | Speech & Audio |
|-------|---------|------------------------|----------------------------|-------------------|------------------|----------------|
| Gemini-3-Pro-Preview | 64.58% | 59.48% | 80.42% | 50.92% | 62.15% | 66.03% |
| GPT-5 | 54.37% | 51.72% | 55.83% | 55.21% | 53.94% | 55.34% |
| Qwen3-Omni | 28.01% | 25.00% | 40.83% | 24.54% | 23.03% | 27.10% |
| Qwen3-VL-32B | 33.94% | 25.86% | 35.00% | 30.67% | 31.23% | 45.42% |
| Vidi2.5-think | 64.33% | 55.17% | 66.25% | 61.35% | 62.78% | 74.43% |
