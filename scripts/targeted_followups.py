#!/usr/bin/env python3
"""Two bounded score-level validation experiments and report evidence export."""
import ast
import collections
import csv
import datetime as dt
import json
import math
import os
import sqlite3
import sys
import time

import numpy as np
from scipy import sparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
PROC = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "results")
os.makedirs(RESULTS, exist_ok=True)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import compare_models as cm
from build_interactions import merged_coverage

ALPHA = 0.4
BETA = 0.3
LOWER_MULTIPLIER = 0.5
UPPER_MULTIPLIER = 2.0
HIST_WEEKS = 10.0
FUTURE_WEEKS = 10.0


def minmax(scores):
    low, high = float(np.min(scores)), float(np.max(scores))
    if not math.isfinite(low) or not math.isfinite(high) or high <= low:
        return np.zeros_like(scores, dtype=np.float32)
    return ((scores - low) / (high - low)).astype(np.float32)


def schedule_program_channel_counts(files, start_date, split_date, end_date):
    """Counts unique labelled airings by programme/channel within fixed windows."""
    train_start = dt.datetime.strptime(start_date, "%Y-%m-%d")
    split = dt.datetime.strptime(split_date, "%Y-%m-%d")
    val_end = dt.datetime.strptime(end_date, "%Y-%m-%d")
    events = {}
    for filename in files:
        with open(os.path.join(RAW, filename), newline="", encoding="utf-8-sig") as source:
            for row in csv.DictReader(source):
                if row["tv_show_id"] == "0" or int(row["duration"]) <= 0:
                    continue
                start = dt.datetime.strptime(row["start_time"], "%d.%m.%Y %H:%M:%S")
                end = start + dt.timedelta(seconds=int(row["duration"]))
                key = (row["channel_id"], start, int(row["duration"]), row["tv_show_id"])
                events[key] = end
    hist = collections.Counter()
    future = collections.Counter()
    candidates = set()
    for (channel, start, duration, pid), end in events.items():
        if max(start, train_start) < min(end, split):
            hist[(pid, channel)] += 1
        if max(start, split) < min(end, val_end):
            future[(pid, channel)] += 1
            candidates.add(pid)
    return hist, future, candidates


def user_channel_preferences(users, phases=("train",)):
    """Read existing overlap segments only; do not rerun view/schedule joining."""
    db = sqlite3.connect(os.path.join(PROC, "overlap_segments.sqlite"))
    placeholders = ",".join("?" for _ in phases)
    sql = ("SELECT user_id,event_key,program_id,duration,seg_start,seg_end "
           "FROM segments WHERE phase IN (" + placeholders + ") "
           "ORDER BY user_id,event_key,seg_start,seg_end")
    query = db.execute(sql, tuple(phases))
    wanted = set(users)
    pref = collections.Counter()
    current_key = None
    pieces = []
    event_info = None

    def flush(key, data, spans):
        if key is None:
            return
        user, event_key = key
        _, channel, _, duration, _ = data
        total, _ = merged_coverage(spans)
        if total / duration >= 0.80:
            pref[(user, channel)] += 1

    for user, event_text, pid, duration, start, end in query:
        if user not in wanted:
            continue
        key = (user, event_text)
        if key != current_key:
            flush(current_key, event_info, pieces)
            current_key = key
            event = ast.literal_eval(event_text)
            channel, event_start, event_duration, event_pid = event
            event_info = (event, channel, event_start, event_duration, event_pid)
            pieces = []
        pieces.append((start, end))
    flush(current_key, event_info, pieces)
    db.close()
    return pref


def channel_multiplier(users, candidates, preferences, hist_counts, future_counts,
                       hist_weeks=HIST_WEEKS, future_weeks=FUTURE_WEEKS):
    channels = sorted({c for _, c in preferences})
    cidx = {c: i for i, c in enumerate(channels)}
    uidx = {u: i for i, u in enumerate(users)}
    pidx = {p: i for i, p in enumerate(candidates)}
    ur, uc, uv = [], [], []
    per_user = collections.defaultdict(list)
    for (u, c), n in preferences.items():
        if u in uidx and c in cidx:
            per_user[u].append((cidx[c], n))
    for u, pairs in per_user.items():
        total = sum(n for _, n in pairs)
        for ci, n in pairs:
            ur.append(uidx[u]); uc.append(ci); uv.append(n / total)
    pref_matrix = sparse.csr_matrix((uv, (ur, uc)), shape=(len(users), len(channels)), dtype=np.float64)
    hr, hc, hv, fr, fc, fv = [], [], [], [], [], []
    for (pid, channel), count in hist_counts.items():
        if pid in pidx and channel in cidx:
            hr.append(pidx[pid]); hc.append(cidx[channel]); hv.append(count / hist_weeks)
    for (pid, channel), count in future_counts.items():
        if pid in pidx and channel in cidx:
            fr.append(pidx[pid]); fc.append(cidx[channel]); fv.append(count / future_weeks)
    hist_rates = sparse.csr_matrix((hv, (hr, hc)), shape=(len(candidates), len(channels)))
    future_rates = sparse.csr_matrix((fv, (fr, fc)), shape=(len(candidates), len(channels)))
    user_hist = (pref_matrix @ hist_rates.T).toarray()
    user_future = (pref_matrix @ future_rates.T).toarray()
    ratios = np.ones_like(user_hist, dtype=np.float64)
    supported = user_hist > 0
    ratios[supported] = user_future[supported] / user_hist[supported]
    ratios = np.clip(ratios, LOWER_MULTIPLIER, UPPER_MULTIPLIER)
    multipliers = np.power(ratios, BETA).astype(np.float32)
    diagnostics = {
        "users_with_qualified_train_channel_history": len(per_user),
        "preferred_channels": len(channels),
        "historical_window_weeks": hist_weeks,
        "future_window_weeks": future_weeks,
        "zero_historical_rate_candidates_neutral_ratio": int((~supported).sum()),
        "ratio_clip": [LOWER_MULTIPLIER, UPPER_MULTIPLIER],
        "beta": BETA,
        "multiplier_min_max": [float(multipliers.min()), float(multipliers.max())],
    }
    return multipliers, diagnostics


def score_rankings(score_map, candidates, fallback):
    fallback_pos = {p: i for i, p in enumerate(fallback)}
    indexes = {p: i for i, p in enumerate(candidates)}
    orders = {}
    for u, scores in score_map.items():
        order = sorted(range(len(candidates)), key=lambda i: (
            -float(scores[i]) if np.isfinite(scores[i]) else math.inf,
            fallback_pos.get(candidates[i], len(fallback)), cm.pid_key(candidates[i])))
        orders[u] = [candidates[i] for i in order]
    return orders


def recs5(orders, users):
    output = {}
    for u in users:
        seen = set()
        picks = []
        for p in orders[u]:
            if p not in seen:
                picks.append(p); seen.add(p)
            if len(picks) == 5:
                break
        output[u] = picks
    return output


def read_title_category(candidates):
    title = {}
    cat = {}
    seen = collections.Counter()
    for filename in ("export_arh_11-20-final.csv", "export_arh_21-30-final.csv"):
        with open(os.path.join(RAW, filename), newline="", encoding="utf-8-sig") as source:
            for row in csv.DictReader(source):
                pid = row["tv_show_id"]
                if pid in candidates and pid != "0":
                    seen[(pid, row["tv_show_title"], row["tv_show_category"])] += 1
    for pid in candidates:
        entries = [(t, c, n) for (p, t, c), n in seen.items() if p == pid]
        if entries:
            t, c, _ = sorted(entries, key=lambda x: (-x[2], x[0], x[1]))[0]
            title[pid], cat[pid] = t, c
        else:
            title[pid], cat[pid] = "[title unavailable]", "[category unavailable]"
    return title, cat


def main():
    clock = time.perf_counter()
    metrics = json.load(open(os.path.join(PROC, "model_comparison.json"), encoding="utf-8"))
    users = cm.read_ids(os.path.join(RAW, "submission.csv"))
    gt_path = os.path.join(PROC, "validation_ground_truth.csv")
    gt = cm.read_ground_truth(gt_path)
    gt_order = collections.defaultdict(list)
    with open(gt_path, newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            gt_order[row["user_id"]].append(row["tv_show_id"])
    train = cm.read_interactions([os.path.join(PROC, "train_interactions.csv")])
    candidates = sorted(cm.schedule_pool(("export_arh_11-20-final.csv", "export_arh_21-30-final.csv"),
                                         "2020-05-18", "2020-07-27"), key=cm.pid_key)
    items = sorted({p for _, p in train}, key=cm.pid_key)
    graph_users = sorted({u for u, _ in train} | set(users))
    matrix, _, _ = cm.build_matrix(train, graph_users, items)
    ppr, ppr_runtime = cm.ppr_scores(matrix, graph_users, users, candidates, items, (ALPHA,))
    ppr_raw = ppr[ALPHA]
    counts = collections.defaultdict(dict)
    popularity_counts = collections.Counter()
    for (u, p), n in train.items():
        counts[u][p] = n
        popularity_counts[p] += n
    popularity_order = sorted(candidates, key=lambda p: (-popularity_counts[p], cm.pid_key(p)))
    popularity_ranked = {u: popularity_order[:5] for u in users}
    logfreq = {}
    fused_a = {}
    for u in users:
        raw_ppr = ppr_raw[u]
        freq = np.asarray([math.log1p(counts[u].get(p, 0)) for p in candidates], dtype=np.float32)
        nppr = minmax(raw_ppr)
        nf = minmax(freq)
        logfreq[u] = freq
        fused_a[u] = 0.8 * nppr + 0.2 * nf

    ppr_order = score_rankings(ppr_raw, candidates, popularity_order)
    a_order = score_rankings(fused_a, candidates, popularity_order)
    ppr_recs = recs5(ppr_order, users)
    a_recs = recs5(a_order, users)
    train_users = {u for u, _ in train}
    ppr_map = cm.map_metrics(ppr_recs, gt, users, train_users)
    a_map = cm.map_metrics(a_recs, gt, users, train_users)

    # Train-channel preference is derived from qualified TRAIN-only overlap segments.
    history_start = time.perf_counter()
    preferences = user_channel_preferences(users)
    hist_schedule, future_schedule, _ = schedule_program_channel_counts(
        ("export_arh_11-20-final.csv", "export_arh_21-30-final.csv"),
        "2020-03-08", "2020-05-18", "2020-07-27")
    multipliers, channel_info = channel_multiplier(users, candidates, preferences, hist_schedule, future_schedule)
    best_base = "score_fusion_A" if a_map["all_1260"] > ppr_map["all_1260"] else "PPR_0.4"
    base_scores = fused_a if best_base == "score_fusion_A" else ppr_raw
    b_scores = {u: base_scores[u] * multipliers[i] for i, u in enumerate(users)}
    b_order = score_rankings(b_scores, candidates, popularity_order)
    b_recs = recs5(b_order, users)
    b_map = cm.map_metrics(b_recs, gt, users, train_users)

    # Restore MF validation scores only because the prior run saved hashes/metrics,
    # not the latent factors or predictions needed for report examples.
    mf_start = time.perf_counter()
    factors = 64
    mf_scores, mf_info = cm.mf_scores(matrix, {u: i for i, u in enumerate(graph_users)},
        {p: i for i, p in enumerate(items)}, users, candidates, items, factors)
    mf_order = score_rankings(mf_scores, candidates, popularity_order)
    mf_recs = recs5(mf_order, users)
    mf_map = cm.map_metrics(mf_recs, gt, users, train_users)
    assert abs(mf_map["all_1260"] - metrics["mf"]["64"]["all_1260"]) < 1e-10

    # Source files are not modified. The fixed PPR candidate remains the final choice unless B wins.
    choices = [("PPR_0.4", ppr_map["all_1260"]), ("score_fusion_A", a_map["all_1260"]),
               ("channel_schedule_B", b_map["all_1260"])]
    winner, winner_score = max(choices, key=lambda x: x[1])
    final_path = os.path.join(ROOT, "submissions", "final_candidate.csv")
    final_refit = {"updated": False, "reason": "PPR_0.4 retained; existing full-history candidate reused"}
    if winner != "PPR_0.4":
        # Refit/re-score only the winning PPR-based recipe on weeks 11-30.
        full_counts = train.copy()
        full_counts.update(cm.read_interactions([os.path.join(PROC, "validation_interactions.csv")]))
        full_users = sorted({u for u, _ in full_counts} | set(users))
        full_items = sorted({p for _, p in full_counts}, key=cm.pid_key)
        full_matrix, _, _ = cm.build_matrix(full_counts, full_users, full_items)
        test_schedule = cm.schedule_pool("export_arh_31-42-final.csv", "2020-07-27", "2020-10-20")
        test_candidates = sorted(test_schedule, key=cm.pid_key)
        full_pop = collections.Counter()
        for (_, pid), n in full_counts.items(): full_pop[pid] += n
        full_fallback = sorted(test_candidates, key=lambda p: (-full_pop[p], cm.pid_key(p)))
        full_ppr, full_ppr_info = cm.ppr_scores(full_matrix, full_users, users, test_candidates,
                                                full_items, (ALPHA,))
        full_base = {}
        for user in users:
            ppr_norm = minmax(full_ppr[ALPHA][user])
            item_counts = np.asarray([math.log1p(full_counts.get((user, p), 0))
                                      for p in test_candidates], dtype=np.float32)
            if winner == "score_fusion_A":
                full_base[user] = 0.8 * ppr_norm + 0.2 * minmax(item_counts)
            else:
                full_base[user] = ppr_norm if best_base == "PPR_0.4" else 0.8 * ppr_norm + 0.2 * minmax(item_counts)
        if winner == "channel_schedule_B":
            full_preferences = user_channel_preferences(users, phases=("train", "validation"))
            full_hist_schedule, full_future_schedule, _ = schedule_program_channel_counts(
                ("export_arh_11-20-final.csv", "export_arh_21-30-final.csv", "export_arh_31-42-final.csv"),
                "2020-03-08", "2020-07-27", "2020-10-20")
            full_multipliers, final_channel_info = channel_multiplier(
                users, test_candidates, full_preferences, full_hist_schedule, full_future_schedule,
                hist_weeks=20.0, future_weeks=12.0)
            for i, user in enumerate(users):
                full_base[user] *= full_multipliers[i]
        final_orders = score_rankings(full_base, test_candidates, full_fallback)
        final_recs = recs5(final_orders, users)
        os.makedirs(os.path.dirname(final_path), exist_ok=True)
        with open(final_path, "w", newline="", encoding="utf-8") as target:
            writer = csv.writer(target); writer.writerow(["user_id", "tv_show_id"])
            for user in users: writer.writerow([user, " ".join(final_recs[user])])
        final_refit = {"updated": True, "method": winner, "full_history": "weeks 11-30",
                       "ppr_alpha": ALPHA, "ppr_runtime_seconds": full_ppr_info[str(ALPHA)]["runtime_seconds"],
                       "candidate_count": len(test_candidates)}
        if winner == "channel_schedule_B": final_refit["schedule_feature"] = final_channel_info
    else:
        test_candidates = sorted(cm.schedule_pool(
            "export_arh_31-42-final.csv", "2020-07-27", "2020-10-20"), key=cm.pid_key)
        final_recs = {}
        with open(final_path, newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source): final_recs[row["user_id"]] = row["tv_show_id"].split()
    assert len(final_recs) == 1260
    assert all(len(final_recs[u]) == 5 and len(set(final_recs[u])) == 5 and
               all(p != "0" and p in set(test_candidates) for p in final_recs[u]) for u in users)
    with open(final_path, newline="", encoding="utf-8") as source:
        final_rows = list(csv.DictReader(source))
    assert [r["user_id"] for r in final_rows] == users
    assert all(r["tv_show_id"].strip() for r in final_rows)

    # Choose high- and low-history profiles from the fixed submission users with validation GT.
    history_n = collections.Counter()
    for (u, _), n in train.items():
        history_n[u] += n
    eligible_profiles = [u for u in users if u in gt and history_n[u] > 0]
    heavy = max(eligible_profiles, key=lambda u: (history_n[u], u))
    light = min(eligible_profiles, key=lambda u: (history_n[u], u))
    title, category = read_title_category(set(candidates))
    popularity = popularity_ranked
    final_test = {}
    with open(final_path, newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            final_test[row["user_id"]] = row["tv_show_id"].split()

    def expand(ids):
        return [{"id": p, "title": title.get(p, "[not in validation schedule]"),
                 "category": category.get(p, "[not in validation schedule]")} for p in ids]

    profiles = []
    for user, label in ((heavy, "heavy_train_history"), (light, "light_train_history")):
        relevant = set(gt[user])
        ranked = ppr_recs[user]
        hit_rank = 0
        precisions = []
        ap_terms = []
        for rank, pid in enumerate(ranked, 1):
            if pid in relevant:
                hit_rank += 1
                precision = hit_rank / rank
                precisions.append({"rank": rank, "id": pid, "precision_at_rank": precision})
                ap_terms.append(precision)
        denom = min(5, len(relevant))
        ap = sum(ap_terms) / denom if denom else 0.0
        profiles.append({
            "profile": label, "user_id": user, "qualified_train_airings": history_n[user],
            "train_distinct_programmes": len(counts[user]),
            "validation_ground_truth_top5": expand(gt_order[user]),
            "validation_predictions": {
                "global_popularity": expand(popularity[user]),
                "PPR_alpha_0_4": expand(ppr_recs[user]),
                "MF_64": expand(mf_recs[user]),
                "final_method_validation_prediction": expand(
                    ppr_recs[user] if winner == "PPR_0.4" else (a_recs[user] if winner == "score_fusion_A" else b_recs[user])),
            },
            "final_candidate_test_top5": final_test.get(user),
            "PPR_AP5_audit": {"recommended_ids_in_rank_order": ranked,
                "relevant_ids": sorted(relevant, key=cm.pid_key), "hits_and_precision_terms": precisions,
                "numerator": sum(ap_terms), "denominator_min_5_relevant": denom, "AP@5": ap},
        })

    evidence = {
        "interaction_scale": {"train_users": 4737, "train_user_program_pairs": 114853,
            "train_qualified_airings": 316416, "validation_users_with_ground_truth": 4643,
            "validation_user_program_pairs": 140486, "validation_qualified_airings": 295044,
            "fixed_submission_users": 1260},
        "environment_and_seed": {"python": "3.9.6", "numpy": np.__version__,
            "scipy": __import__("scipy").__version__, "seed": 42,
            "mf_factors": 64, "mf_iterations": 8, "ppr_alpha": ALPHA},
        "join_diagnosis": {
            "initial_zero_join_cause": "The earlier search used bisect_left(starts, view_end), starting at intervals whose starts were at/after the view end. Correct search uses prefix maximum ends to find the earliest possible overlap.",
            "initial_zero_result_was_false": "Corrected full scan found overlaps; no time shift was applied.",
            "all_view_rows_scanned": 5821987,
            "common_view_schedule_channels": 197,
            "views_on_common_channels": 5708408,
            "positive_overlap_any_airtime_view_rows": 5408598,
            "positive_overlap_labelled_view_rows": 2964346,
            "positive_overlap_labelled_user_program_pairs": 1371685,
            "manual_example": {"user_id": "7981779138282146703", "channel": "1495",
                "program_id": "12001732", "overlap_seconds": 1380, "programme_duration_seconds": 1380},
            "diagnostic_file": "join_diagnostic.txt",
        },
        "experiment_A": {"formula": "0.8*per-user-minmax(PPR_alpha_0.4)+0.2*per-user-minmax(log1p(TRAIN user-program count))",
                          "zero_range": "normalized vector is all zero; global popularity is ranking tie fallback",
                          "MAP@5": a_map},
        "experiment_B": {"formula": "base_score * clip((future_airings/week / historical_airings/week)^0.3, 0.5, 2.0)",
                          "base": best_base, "MAP@5": b_map, "channel_diagnostics": channel_info,
                          "history_source": "qualified TRAIN-only segments in existing overlap_segments.sqlite",
                          "future_schedule_source": "validation-period schedule only; no validation/test viewing behavior"},
        "reference_PPR": ppr_map,
        "selected_winner": winner,
        "selected_MAP@5": winner_score,
        "final_candidate_refit": final_refit,
        "mf_validation_reproduction": {"MAP@5": mf_map, "factor_count": factors,
            "runtime_seconds": mf_info["runtime_seconds"], "matches_saved_metrics": True,
            "note": "Latent factors/prediction vectors were not saved in original run; one fixed-seed fit was rerun solely to restore required example predictions, not to sweep."},
        "ppr_score_recomputation_seconds": ppr_runtime["0.4"]["runtime_seconds"],
        "channel_history_derivation_seconds": round(time.perf_counter()-history_start-mf_info["runtime_seconds"], 3),
        "targeted_workflow_seconds": round(time.perf_counter()-clock, 3),
        "user_profiles": profiles,
        "join_example": {"user_id": "7981779138282146703", "vsetv_id/channel_id": "1495",
            "view_interval": ["2020-03-09 00:40:50", "2020-03-09 02:28:55"],
            "programme_id": "12001732", "programme_interval": ["2020-03-09 01:16:00", "2020-03-09 01:39:00"],
            "programme_duration_seconds": 1380, "overlap_seconds": 1380, "watched_fraction": 1.0},
        "prior_model_table": metrics,
        "submission_status": "not submitted to Kaggle",
        "repository_readiness": {"readme": "README.md", "requirements": "requirements.txt",
            "source_assignment": "exam_quiz_AVITO.md", "final_candidate": "submissions/final_candidate.csv",
            "raw_and_processed_data_gitignored": True, "github_published": False},
    }
    with open(os.path.join(RESULTS, "targeted_experiments_and_examples.json"), "w", encoding="utf-8") as f:
        json.dump(evidence, f, ensure_ascii=False, indent=2)
    with open(os.path.join(RESULTS, "model_comparison.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f); w.writerow(["method", "MAP@5_all_1260", "MAP@5_eligible_1224"])
        all_methods = [("Global popularity", metrics["baselines"]["global_popularity"]),
            ("Personal repeat", metrics["baselines"]["personal_repeat_frequency"])]
        all_methods += [("PPR alpha="+k,v) for k,v in metrics["ppr"].items()]
        all_methods += [("MF factors="+k,v) for k,v in metrics["mf"].items()]
        all_methods += [(k,v) for k,v in metrics["hybrids"].items()]
        all_methods += [("Score fusion A", a_map), ("Channel schedule B", b_map)]
        for name, value in all_methods:
            w.writerow([name, value.get("all_1260", value.get("MAP@5_all_1260")),
                        value.get("eligible_1224", value.get("MAP@5_users_with_train_history_and_validation_ground_truth"))])
    print(json.dumps({"PPR": ppr_map, "A": a_map, "B": b_map, "winner": winner,
                      "mf_restore_seconds": mf_info["runtime_seconds"],
                      "profiles": [(x["profile"],x["user_id"],x["qualified_train_airings"]) for x in profiles],
                      "evidence": os.path.join(RESULTS,"targeted_experiments_and_examples.json")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
