# Migrating from VUE-PLOT to CineMME

The legacy `VUE_PLOT/` directory has been replaced by `CineMME/`. Run all
commands from `CineMME/`; see [README.md](README.md) for setup and
[evaluation/README.md](evaluation/README.md) for inference and scoring.

## Files and evaluation

Previous paths below are relative to `VUE_PLOT/`; replacement paths are relative
to `CineMME/`.

| Previous VUE-PLOT file | CineMME replacement |
| --- | --- |
| `Character_Grounding/VUE-PLOT_Character_ground_truth.json` and `VUE-PLOT_Character_ground_truthwith_path.json` | `datasets/cinemme_grounding.json` (546 videos) |
| `Reasoning_VQA/VUE-PLOT_Reasoning_VQA.json` | `datasets/cinemme_reasoning_mcq.json` (1,600 questions) |
| No open-ended question file | `datasets/cinemme_reasoning_oeq.json` (316 questions, including judge context) |
| `character_eval.py` | `python -m evaluation.scoring.score_grounding` |
| `vqa_eval.py` | `python -m evaluation.scoring.score_mcq` |

The previous MCQ release contained 1,214 records. CineMME uses `question_id`,
`question`, and structured options with `label` and `text` fields. Question IDs
and option letters have changed. Match questions by video ID and question text,
then align option text before reusing historical predictions; copying answer
letters across versions can change their meaning. The scoring guide documents
the new prediction schemas and coverage denominators. Historical VUE-PLOT
results remain available in Git history and are not scores for this release.

## Reference answer for issue #36

[Vidi issue #36](https://github.com/bytedance/vidi/issues/36) identified a string
`"None"` reference answer for legacy `problem_id: 193`, video `6MIGRX1AVKo`.
The question asks about voices singing "Fall in line" in different acoustic
spaces between 2:31 and 2:35.

The corresponding CineMME record is `mcq_000959` in
`datasets/cinemme_reasoning_mcq.json`, with `answer: "B"`. The question and all
six option texts are unchanged, but their order differs. CineMME option B is
identical to legacy option D: the voices reinforce the montage and express the
pervasive command to conform throughout Heaven. The new dataset already contains
this corrected reference; this migration preserves it.

## Annotation revisions

See [datasets/README.md](datasets/README.md) for schemas, validation, version
notes, and known annotation limitations. Use the dataset version and option
order that generated each prediction when computing scores.
