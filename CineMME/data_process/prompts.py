"""Prompt templates used in the public CineMME construction pipeline."""

from __future__ import annotations

from string import Template


VIDEO_ANALYSIS_PROMPT = Template("""You are a multimodal film annotator building gold metadata for a movie-understanding benchmark.

Inputs:
- Video clip with audio.
- Movie/cast metadata:
$metadata
- Face tracks with normalized bboxes:
$face_tracks
- ASR chunks:
$asr_chunks

Tasks:
1. Character identification: filter noisy face tracks, identify relevant characters when possible, and assign "unknown" when evidence is insufficient.
2. Speaker diarization: clean ASR, split chunks when multiple speakers occur, and align each speech segment to the active face ID. Use -1 for off-screen, occluded, or ambiguous speakers.
3. Narrative and plot segmentation: summarize literal actions and narrative function over short temporal windows.

Rules:
- Do not invent dialogue or timestamps.
- Keep timestamps in seconds.
- Cite concrete visual, audio, and contextual evidence in thought fields.
- Metadata IDs are construction aids; they should not be exposed in final benchmark questions.

Return strict JSON following the supplied schema.
""")


QA_GENERATION_PROMPT = Template("""You are a film analyst and assessment designer creating high-difficulty CineMME MCQs.

Inputs:
- Movie/cast metadata:
$metadata
- ASR chunks:
$asr_chunks
- Video analysis JSON:
$video_analysis

Generate exactly $num_questions six-option, single-answer multiple-choice questions.

Main categories:
1. Perception: fine-grained visual/audio facts, character/object localization, counting, or recognition.
2. Audio: dialogue wording, vocal delivery, subtle sound effects, off-screen audio, acoustic space.
3. Narrative: causality, chronology, foreshadowing, plot structure, information asymmetry.
4. Social: intentions, relationships, beliefs, emotion shifts, commonsense or knowledge integration.
5. Cinematic: camera, editing, shot composition, sound design, lighting, props, mise-en-scene.

Design rules:
- Each question must require a concrete audiovisual key from a narrow time window.
- A non-viewer should have near-random chance from the stem and options alone.
- Distractors must be plausible shortcuts but contradicted by the clip.
- Options A-F must be balanced in length, tone, and specificity.
- Explanations must cite approximate timestamps and falsify distractors.

Return strict JSON following the supplied QA schema.
""")


DISTRACTOR_REFINEMENT_PROMPT = Template("""You are refining CineMME distractors after model probing.

Input items:
$items

Rewrite weak distractors only. Keep the original correct answer.

Rules:
- Create plausible but wrong shortcuts based on visual salience, audio priors, narrative tropes, social assumptions, or cinematic terminology.
- Do not make the correct answer stand out by length, detail, or wording.
- Every revised distractor must remain contradicted by video/audio evidence.
- Return the same item IDs and exactly five distractors for each item.

Return strict JSON following the supplied schema.
""")


OEQ_CONVERSION_PROMPT = Template("""Convert the verified MCQ into an open-ended CineMME question.

MCQ item:
$item

Rules:
- Remove option letters and option-specific wording.
- Preserve the same audiovisual evidence anchor.
- Provide a concise reference answer.
- Provide a reasoning trace that cites the key visual, audio, dialogue, and cinematic evidence.

Return JSON with question, reference_answer, and reasoning_trace.
""")


GROUNDING_PROMPT = """Transcribe all active speakers and track the active speaker face.
Timestamp format: total seconds as a float.
Bounding box format: [x0, y0, x1, y1], normalized to [0, 1].
Sampling strategy: include the first visible frame and each following integer second within the speech window.
Return only a JSON object with an `outputs` list. Do not include explanations.
"""
