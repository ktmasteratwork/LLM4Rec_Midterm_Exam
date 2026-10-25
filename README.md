# sweet.tv TV Programme Recommender — LLM4Rec Final Exam

**Kaggle competition:** https://www.kaggle.com/competitions/sweettv-tv-program-recommender  
**Final Kaggle submission:** [`submissions/final_candidate.csv`](submissions/final_candidate.csv)  
**Project:** offline recommendation pipeline (no web app / no deployed demo)  
**Course:** LLM4Rec — AI-Assisted Recommender Systems, HSE University  
**Student:** Ekaterina Zueva · **Individual submission** · October 2026

## Overview

The goal is to rank five TV programmes for each of 1,260 Kaggle test users. View intervals are matched to the broadcast schedule by channel and overlapping time windows to obtain implicit user–programme feedback. Schedule entries with `tv_show_id = 0` cannot be treated as programme items.

The project uses a chronological validation: training on weeks 11–20, evaluation on weeks 21–30, and a final refit on weeks 11–30 before generating predictions for weeks 31–42. A programme is considered watched when the union of viewing intervals covers at least 80% of its airing duration. Repeated programmes are eligible for recommendation; users with no usable history receive a training-popularity fallback.

## Methods

- **Popularity baseline:** number of qualifying training-period programme airings.
- **Personal repeat baseline:** programme viewing counts in a user's training history.
- **Personalized PageRank (PPR):** user–programme graph weighted by qualifying airing counts, with programme-history restart distribution.
- **Implicit-feedback matrix factorization (MF):** weighted alternating least squares (ALS), with logged view counts as observed targets and confidence weights.
- **Exploratory hybrids:** reciprocal-rank fusion and normalized score fusion.
- **Final model:** PPR with restart probability `α = 0.4`, followed by channel-aware schedule re-ranking with `β = 0.3`. Future **broadcast schedules**, not future user views, supply availability and frequency features.

## Results

MAP@5 uses the same fixed 1,260-user validation cohort for all compared methods.

| Method | Validation MAP@5 |
| --- | ---: |
| Global popularity | 0.29096 |
| Personal repeat frequency | 0.36217 |
| MF, 64 latent factors | 0.35997 |
| PPR, restart α = 0.4 | 0.37961 |
| PPR + channel-aware schedule re-ranking (final) | **0.38539** |

For the uploaded `final_candidate.csv`, Kaggle returned:

| Score | Value |
| --- | ---: |
| Public MAP@5 | 0.42235 |
| **Private MAP@5** | **0.41169** |

Submission status: **Complete (after deadline)**. The old Kaggle leaderboard is frozen; the exam uses the returned private score, not historical leaderboard position.

See [`results/model_comparison.csv`](results/model_comparison.csv) for the complete validation comparison and [`results/validation_examples.json`](results/validation_examples.json) for two validation-user examples, ground truth, programme titles/categories and manual AP@5 breakdowns. Evaluation on 1,224 users with both qualifying training history and nonempty validation ground truth is reported separately.

## Repository files

- [`submissions/final_candidate.csv`](submissions/final_candidate.csv) — **exact final Kaggle prediction file** (1,260 rows, five unique programme IDs per user).
- [`results/model_comparison.csv`](results/model_comparison.csv) — validation metrics of baselines, individual models and hybrids.
- [`scripts/diagnose_join.py`](scripts/diagnose_join.py) — independent streaming diagnosis of channel/time overlaps.
- [`scripts/build_interactions.py`](scripts/build_interactions.py) — generate temporal interactions, 80% coverage labels, ground truth and baselines.
- [`scripts/compare_models.py`](scripts/compare_models.py) — train/evaluate PPR, MF and initial hybrids; build an initial candidate.
- [`scripts/targeted_followups.py`](scripts/targeted_followups.py) — run two bounded re-ranking experiments, refit the chosen model and prepare final submission/evidence.
- `requirements.txt` — Python environment dependencies.
- `exam_quiz_AVITO.md` — provided exam specification.

## Environment and data

Python 3.9.6 · NumPy 1.26.4 · SciPy 1.13.1 · random seed 42 · Apple M2, 16 GiB RAM.

Download the sweet.tv CSV files from the official Kaggle competition, after accepting its rules, into `data/raw/`. The source data, derived interaction matrices, Python virtual environment, cached outputs, session export and credentials are **not** published in this repository.

Place the following competition data files in `data/raw/` with their original names: `dataset11-30.csv`, `export_arh_11-20-final.csv`, `export_arh_21-30-final.csv`, `export_arh_31-42-final.csv`, `submission.csv`.

From the repository root, create a virtual environment and run the scripts in order:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

python scripts/diagnose_join.py         # optional diagnostic
python scripts/build_interactions.py
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/compare_models.py
OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 python scripts/targeted_followups.py
```

The scripts write derived files into `data/processed/` and predictions into `submissions/`; re-running can replace the existing `final_candidate.csv`. Keep a copy of the published submission if reproducing experiments. These commands require the raw Kaggle data and may take time. This repository provides the implementation and recorded evidence; it does not bundle the competition dataset.

## Verification and limitations

Independent checks were performed on the interval-overlap calculation and AP@5 for two example users: a 1,380-second full-airing overlap (100% coverage), and AP@5 values of 0.60 and 0.76. A mistaken interval-search boundary originally produced zero matches; the join was corrected and rerun on the full logs. Most broadcast schedule rows have no valid programme ID, and the single hold-out is not an online/A-B evaluation. Any claims of user satisfaction or revenue impact would be unsupported.

## AI usage

OpenCode was used for coding, model experiments and programmatic checks; ChatGPT assisted with planning, review and prompts. The student personally recomputed the overlap and both AP@5 examples. The full unmodified AI-tool session is submitted separately through the course channel (not stored on GitHub).
