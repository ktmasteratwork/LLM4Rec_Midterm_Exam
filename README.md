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

See [`results/model_comparison.csv`](results/model_comparison.csv) for the complete validation comparison. Evaluation on 1,224 users with both qualifying training history and nonempty validation ground truth is reported separately.

## Repository files

- [`submissions/final_candidate.csv`](submissions/final_candidate.csv) — **exact final Kaggle prediction file** (1,260 rows, five unique programme IDs per user).
- [`results/model_comparison.csv`](results/model_comparison.csv) — validation metrics of baselines, individual models and hybrids.
- `scripts/` — dataset-join diagnostics, interaction construction, PPR/MF evaluation and final prediction code (see source tree).
- `requirements.txt` — Python environment dependencies.
- `exam_quiz_AVITO.md` — provided exam specification.

## Environment and data

Python 3.9.6 · NumPy 1.26.4 · SciPy 1.13.1 · random seed 42 · Apple M2, 16 GiB RAM.

Download the sweet.tv CSV files from the official Kaggle competition, after accepting its rules, into `data/raw/`. The source data, derived interaction matrices, Python virtual environment, cached outputs, session export and credentials are **not** published in this repository.

The implementation constructs chronological train and validation interactions from the original viewing log and schedules, evaluates all methods with the same MAP@5 code and candidate pool, and refits the selected model before generating the Kaggle CSV. See the source scripts for the exact execution entry points and options.

## Verification and limitations

Independent checks were performed on the interval-overlap calculation and AP@5 for two example users: a 1,380-second full-airing overlap (100% coverage), and AP@5 values of 0.60 and 0.76. A mistaken interval-search boundary originally produced zero matches; the join was corrected and rerun on the full logs. Most broadcast schedule rows have no valid programme ID, and the single hold-out is not an online/A-B evaluation. Any claims of user satisfaction or revenue impact would be unsupported.

## AI usage

OpenCode was used for coding, model experiments and programmatic checks; ChatGPT assisted with planning, review and prompts. The student personally recomputed the overlap and both AP@5 examples. The full unmodified AI-tool session is submitted separately through the course channel (not stored on GitHub).
