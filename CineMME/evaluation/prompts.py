"""Evaluation prompt templates."""

from __future__ import annotations

from string import Template


MCQ_PROMPT = Template("""You are a helpful video understanding assistant that answers multi-choice questions through step-by-step reasoning based on the video.

# Instructions:
1. Break down the reasoning process into clear, specific events.
2. Conclude with the best option letter from the provided options at last.

# Output Format:
<Step 1> Description of event/observation
<Step 2> Description of event/observation
...
<Answer> Option letter

Question:
$question

Options:
$options
""")


OEQ_PROMPT = Template("""You are a helpful video understanding assistant that answers Open-end questions through step-by-step reasoning based on the video.

# Instructions:
1. Break down the reasoning process into clear, specific events.
2. Conclude with the best answer at last.

# Output Format:
<Step 1> Description of event/observation
<Step 2> Description of event/observation
...
<Answer> Your Answer

Question:
$question
""")


GROUNDING_PROMPT = """Transcribe all active speaker's speech and track their face using bounding boxes.
Timestamp format: Total seconds (float).
Bounding box format: [x0, y0, x1, y1] with values in [0, 1] normalized to the video frame, tightly enclosing the speaker's face.
Sampling strategy: Sample at the first visible frame and every subsequent integer second.
Output only a single JSON object and do not output explanation.
"""

# Hosted-API variant with an inline format example, used with JSON-object mode for API
# models; kept verbatim, including the doubled braces.
GROUNDING_PROMPT_API = """Transcribe all active speaker's speech and track their face
using bounding boxes. Output only a single JSON object and
do not output explanation.
Timestamp format: Total seconds (float).
Bounding box format: [x0, y0, x1, y1] with values in [0, 1] normalized
to the video frame, tightly enclosing the speaker's face.
Sampling strategy: Sample at the first visible frame and every subsequent
integer second.
Generate a single JSON object follow the schema.
Example:
{{
    "outputs": [
{{
"text": "Yes.",
  "start": 2.4,
  "end": 3.0,
  "boxes": [
    {"timestamp": 2.4, "box_2d": [0.100, 0.050, 0.200, 0.250]},
    {"timestamp": 3.0, "box_2d": [0.105, 0.035, 0.205, 0.245]}}
  ]
}},
{{
  "text": "Hello everyone.",
  "start": 62.4,
  "end": 65.0,
  "boxes": [
    {"timestamp": 62.4, "box_2d": [0.400, 0.150, 0.600, 0.350]},
    {"timestamp": 63.0, "box_2d": [0.405, 0.155, 0.605, 0.355]},
    {"timestamp": 64.0, "box_2d": [0.415, 0.145, 0.603, 0.325]}
  ]
}},
{{...}}
]
}}
"""

JSON_SCHEMA_FOR_GROUNDING = {
  "type": "object",
  "properties": {
    "outputs": {
      "type": "array",
      "items": {
        "type": "object",
        "properties": {
          "text": {
            "type": "string"
          },
          "start": {
            "type": "number"
          },
          "end": {
            "type": "number"
          },
          "boxes": {
            "type": "array",
            "items": {
              "type": "object",
              "properties": {
                "timestamp": {
                  "type": "number"
                },
                "box_2d": {
                  "type": "array",
                  "items": {
                    "type": "number"
                  }
                }
              },
              "propertyOrdering": [
                "timestamp",
                "box_2d"
              ],
              "required": [
                "timestamp",
                "box_2d"
              ]
            }
          }
        },
        "propertyOrdering": [
          "text",
          "start",
          "end",
          "boxes"
        ],
        "required": [
          "text",
          "start",
          "end",
          "boxes"
        ]
      }
    }
  },
  "propertyOrdering": [
    "outputs"
  ],
  "required": [
    "outputs"
  ]
}
