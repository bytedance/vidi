"""Structured-output schemas for CineMME data construction."""

VIDEO_ANALYSIS_SCHEMA = {
    "type": "object",
    "properties": {
        "character_identification": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "face_id": {"type": "integer"},
                    "character_name": {"type": "string"},
                    "actor_name": {"type": "string"},
                    "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
                    "referring_expression": {"type": "string"},
                    "thought": {"type": "string"},
                },
                "required": ["face_id", "character_name", "actor_name", "confidence", "referring_expression"],
            },
        },
        "speaker_diarization": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "face_id": {"type": "integer"},
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "thought": {"type": "string"},
                },
                "required": ["text", "face_id", "start", "end"],
            },
        },
        "narrative_and_plot_segments": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "plot": {"type": "string"},
                    "narrative": {"type": "string"},
                    "thought": {"type": "string"},
                },
                "required": ["start", "end", "plot", "narrative"],
            },
        },
    },
    "required": ["character_identification", "speaker_diarization", "narrative_and_plot_segments"],
}


QA_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "questions_and_answers": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {"type": "string"},
                    "subtype": {"type": "string"},
                    "question": {"type": "string"},
                    "options": {
                        "type": "object",
                        "properties": {letter: {"type": "string"} for letter in "ABCDEF"},
                        "required": list("ABCDEF"),
                    },
                    "answer": {"type": "string", "enum": list("ABCDEF")},
                    "explanation": {"type": "string"},
                },
                "required": ["type", "subtype", "question", "options", "answer", "explanation"],
            },
        }
    },
    "required": ["questions_and_answers"],
}


DISTRACTOR_REFINEMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "distractors": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "new_distractors": {"type": "array", "items": {"type": "string"}},
                    "notes": {"type": "string"},
                },
                "required": ["id", "new_distractors"],
            },
        }
    },
    "required": ["distractors"],
}


GROUNDING_OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "outputs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "start": {"type": "number"},
                    "end": {"type": "number"},
                    "boxes": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "timestamp": {"type": "number"},
                                "box_2d": {"type": "array", "items": {"type": "number"}},
                            },
                            "required": ["timestamp", "box_2d"],
                        },
                    },
                },
                "required": ["text", "start", "end", "boxes"],
            },
        }
    },
    "required": ["outputs"],
}
