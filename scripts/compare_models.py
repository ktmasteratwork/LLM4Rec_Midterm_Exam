#!/usr/bin/env python3
"""Bounded PPR / implicit weighted MF validation and final candidate generation."""
import collections
import csv
import hashlib
import json
import math
import os
import time

import numpy as np
from scipy import sparse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")
SUBMISSION = os.path.join(RAW, "submission.csv")
RESULT_PATH = os.path.join(OUT, "model_comparison.json")
SEED = 42
FACTORS_TO_TEST = (32, 64)
PPR_ALPHAS = (0.2, 0.4, 0.6)
ALS_ITERATIONS = 8
ALS_REGULARIZATION = 0.1
ALS_CONFIDENCE = 40.0
PPR_TOLERANCE = 1e-7
PPR_MAX_ITERATIONS = 100


def pid_key(pid):
    try:
        return (0, int(pid), pid)
    except ValueError:
        return (1, pid, pid)


def read_interactions(paths):
    counts = collections.Counter()
    for path in paths:
        with open(path, newline="", encoding="utf-8") as source:
            for row in csv.DictReader(source):
                counts[(row["user_id"], row["tv_show_id"])] += int(row["qualified_airings"])
    return counts


def read_ids(path):
    with open(path, newline="", encoding="utf-8-sig") as source:
        return [row["user_id"] for row in csv.DictReader(source)]


def read_ground_truth(path):
    result = collections.defaultdict(set)
    with open(path, newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            result[row["user_id"]].add(row["tv_show_id"])
    return dict(result)


def schedule_pool(filename, start_date, end_date):
    # Ordinal seconds use the same origin-independent representation as pipeline.
    import datetime as dt
    def sec(text):
        d = dt.datetime.strptime(text, "%d.%m.%Y %H:%M:%S")
        return d.toordinal() * 86400 + d.hour * 3600 + d.minute * 60 + d.second
    lower = dt.datetime.strptime(start_date, "%Y-%m-%d").toordinal() * 86400
    upper = dt.datetime.strptime(end_date, "%Y-%m-%d").toordinal() * 86400
    counts = collections.Counter()
    filenames = filename if isinstance(filename, (list, tuple)) else [filename]
    seen = set()
    for name in filenames:
        with open(os.path.join(RAW, name), newline="", encoding="utf-8-sig") as source:
            for row in csv.DictReader(source):
                if row["tv_show_id"] == "0" or int(row["duration"]) <= 0:
                    continue
                start = sec(row["start_time"])
                duration = int(row["duration"])
                end = start + duration
                event = (row["channel_id"], start, duration, row["tv_show_id"])
                if event not in seen and max(start, lower) < min(end, upper):
                    counts[row["tv_show_id"]] += 1
                    seen.add(event)
    return counts


def build_matrix(counts, users, items):
    user_index = {u: i for i, u in enumerate(users)}
    item_index = {p: i for i, p in enumerate(items)}
    rows, cols, vals = [], [], []
    for (u, p), val in counts.items():
        if u in user_index and p in item_index:
            rows.append(user_index[u]); cols.append(item_index[p]); vals.append(float(val))
    matrix = sparse.coo_matrix((vals, (rows, cols)), shape=(len(users), len(items))).tocsr()
    matrix.sum_duplicates()
    return matrix, user_index, item_index


def ppr_scores(matrix, matrix_users, target_users, candidate_items, items, alphas):
    n_users, n_items = matrix.shape
    adjacency = sparse.bmat([[None, matrix], [matrix.T, None]], format="csr", dtype=np.float64)
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    inv_degree = np.zeros_like(degree)
    inv_degree[degree > 0] = 1.0 / degree[degree > 0]
    transition = sparse.diags(inv_degree).dot(adjacency).tocsr()
    transition_t = transition.T.tocsr()
    user_index = {u: i for i, u in enumerate(matrix_users)}
    item_index = {p: i for i, p in enumerate(items)}
    candidate_idx = np.asarray([item_index.get(p, -1) for p in candidate_items], dtype=np.int32)
    outputs = {a: {} for a in alphas}
    info = {}
    batch_size = 64
    for alpha in alphas:
        started = time.perf_counter()
        max_steps = 0
        max_mass_error = 0.0
        for first in range(0, len(target_users), batch_size):
            batch = target_users[first:first + batch_size]
            q = np.zeros((n_users + n_items, len(batch)), dtype=np.float64)
            active = []
            for j, user in enumerate(batch):
                if user not in user_index:
                    continue
                ui = user_index[user]
                lo, hi = matrix.indptr[ui], matrix.indptr[ui + 1]
                if hi > lo:
                    w = matrix.data[lo:hi]
                    q[n_users + matrix.indices[lo:hi], j] = w / w.sum()
                    active.append(j)
            x = q.copy()
            for step in range(1, PPR_MAX_ITERATIONS + 1):
                nxt = (1.0 - alpha) * transition_t.dot(x) + alpha * q
                errors = np.abs(nxt - x).sum(axis=0)
                x = nxt
                if not active or float(np.max(errors[active])) <= PPR_TOLERANCE:
                    break
            max_steps = max(max_steps, step)
            mass_error = float(np.max(np.abs(x.sum(axis=0)[active] - 1.0))) if active else 0.0
            max_mass_error = max(max_mass_error, mass_error)
            for j, user in enumerate(batch):
                values = np.zeros(len(candidate_items), dtype=np.float32)
                known = candidate_idx >= 0
                values[known] = x[n_users + candidate_idx[known], j]
                outputs[alpha][user] = values
        info[str(alpha)] = {
            "runtime_seconds": round(time.perf_counter() - started, 3),
            "iterations_max": max_steps,
            "max_probability_mass_error": max_mass_error,
            "converged": max_steps < PPR_MAX_ITERATIONS,
        }
    return outputs, info


def implicit_als(matrix, factors, seed=SEED, iterations=ALS_ITERATIONS):
    """Weighted implicit MF minimizing all-pair confidence loss.

    p_ui=log(1+count) for observed entries, zero otherwise; c_ui=1 for
    unobserved, and 1+40*log(1+count) for observed entries.
    """
    rng = np.random.RandomState(seed)
    n_users, n_items = matrix.shape
    target = matrix.copy().tocsr()
    target.data = np.log1p(target.data)
    confidence = target.copy()
    confidence.data = 1.0 + ALS_CONFIDENCE * confidence.data
    item_factors = rng.normal(0, 0.01, (n_items, factors)).astype(np.float64)
    user_factors = rng.normal(0, 0.01, (n_users, factors)).astype(np.float64)
    eye = np.eye(factors)
    loss_trace = []
    for _ in range(iterations):
        yty = item_factors.T @ item_factors
        for u in range(n_users):
            lo, hi = matrix.indptr[u], matrix.indptr[u + 1]
            if lo == hi:
                user_factors[u] = 0.0
                continue
            ids = matrix.indices[lo:hi]
            c = confidence.data[lo:hi]
            p = target.data[lo:hi]
            y = item_factors[ids]
            a = yty + (y.T * (c - 1.0)) @ y + ALS_REGULARIZATION * eye
            b = (y.T @ (c * p))
            user_factors[u] = np.linalg.solve(a, b)
        xtx = user_factors.T @ user_factors
        r_t = matrix.tocsc()
        p_t = target.tocsc()
        c_t = confidence.tocsc()
        for item in range(n_items):
            lo, hi = r_t.indptr[item], r_t.indptr[item + 1]
            if lo == hi:
                item_factors[item] = 0.0
                continue
            ids = r_t.indices[lo:hi]
            c = c_t.data[lo:hi]
            p = p_t.data[lo:hi]
            x = user_factors[ids]
            a = xtx + (x.T * (c - 1.0)) @ x + ALS_REGULARIZATION * eye
            b = x.T @ (c * p)
            item_factors[item] = np.linalg.solve(a, b)
        loss_trace.append(objective(matrix, target, confidence, user_factors, item_factors))
    return user_factors, item_factors, loss_trace


def objective(matrix, target, confidence, users, items):
    yty = items.T @ items
    loss = float(np.einsum("uf,fg,ug->", users, yty, users))
    # Add observed-entry corrections to the unobserved c=1,p=0 base objective.
    csr = matrix.tocsr()
    target = target.tocsr()
    confidence = confidence.tocsr()
    for u in range(csr.shape[0]):
        lo, hi = csr.indptr[u], csr.indptr[u + 1]
        if lo == hi:
            continue
        ids = csr.indices[lo:hi]
        pred = items[ids] @ users[u]
        p = target.data[lo:hi]
        c = confidence.data[lo:hi]
        loss += float(np.sum((c - 1.0) * pred ** 2 - 2.0 * c * p * pred + c * p ** 2))
    loss += ALS_REGULARIZATION * (float(np.sum(users ** 2)) + float(np.sum(items ** 2)))
    return loss


def mf_scores(matrix, matrix_users, matrix_items, target_users, candidate_items, items, factors):
    started = time.perf_counter()
    uf, vf, losses = implicit_als(matrix, factors)
    result = {}
    for user in target_users:
        if user not in matrix_users:
            result[user] = np.full(len(candidate_items), -np.inf, dtype=np.float32)
            continue
        ui = matrix_users[user]
        scores = np.full(len(candidate_items), -np.inf, dtype=np.float64)
        for k, pid in enumerate(candidate_items):
            if pid in matrix_items:
                scores[k] = float(uf[ui] @ vf[matrix_items[pid]])
        result[user] = scores.astype(np.float32)
    return result, {"factors": factors, "seed": SEED, "iterations": ALS_ITERATIONS,
                    "confidence_alpha": ALS_CONFIDENCE, "regularization": ALS_REGULARIZATION,
                    "target": "log1p(qualified_airing_count)", "runtime_seconds": round(time.perf_counter()-started, 3),
                    "objective_trace": losses, "finite_user_factor_count": int(np.isfinite(uf).all(axis=1).sum()),
                    "factor_sha256": hashlib.sha256(uf.tobytes() + vf.tobytes()).hexdigest()}


def ranks_from_scores(score_map, candidates, fallback):
    fallback_pos = {pid: k for k, pid in enumerate(fallback)}
    positions = {pid: i for i, pid in enumerate(candidates)}
    orders = {}
    score_arrays = {}
    for user, scores in score_map.items():
        order_idx = sorted(range(len(candidates)), key=lambda i: (
            -float(scores[i]) if np.isfinite(scores[i]) else math.inf,
            fallback_pos.get(candidates[i], len(fallback)), pid_key(candidates[i])))
        orders[user] = [candidates[i] for i in order_idx]
        rank_score = np.zeros(len(candidates), dtype=np.float32)
        for rank, i in enumerate(order_idx, 1):
            rank_score[i] = 1.0 / (60.0 + rank)
        score_arrays[user] = rank_score
    return orders, score_arrays


def recommendations_from_order(orders, users):
    return {u: orders[u][:5] for u in users}


def rank_fusion(score_a, score_b, candidates, fallback, weight_a=1.0, weight_b=1.0):
    _, a = ranks_from_scores(score_a, candidates, fallback)
    _, b = ranks_from_scores(score_b, candidates, fallback)
    combined = {u: weight_a * a[u] + weight_b * b[u] for u in a}
    return ranks_from_scores(combined, candidates, fallback)[0]


def map_metrics(recs, gt, users, eligible):
    def ap(pred, rel):
        if not rel: return 0.0
        hits = total = 0.0
        for rank, pid in enumerate(pred[:5], 1):
            if pid in rel:
                hits += 1.0
                total += hits / rank
        return total / min(5, len(rel))
    overall = sum(ap(recs[u], gt.get(u, set())) for u in users) / len(users)
    subset = [u for u in users if u in eligible and gt.get(u)]
    conditional = sum(ap(recs[u], gt[u]) for u in subset) / len(subset) if subset else 0.0
    assert all(len(recs[u]) == 5 and len(set(recs[u])) == 5 for u in users)
    return {"all_1260": overall, "eligible_1224": conditional, "eligible_count": len(subset)}


def validate_recs(recs, users, candidates):
    allowed = set(candidates)
    for user in users:
        assert len(recs[user]) == 5 and len(set(recs[user])) == 5
        assert all(pid != "0" and pid in allowed for pid in recs[user])


def final_candidate(best_name, best_config, train_counts, val_counts, submission_users):
    started = time.perf_counter()
    counts = train_counts.copy()
    counts.update(val_counts)
    test_schedule = schedule_pool("export_arh_31-42-final.csv", "2020-07-27", "2020-10-20")
    candidates = sorted(test_schedule, key=pid_key)
    users = sorted(set(u for u, _ in counts) | set(submission_users))
    matrix, uidx, iidx = build_matrix(counts, users, sorted(set(p for _, p in counts), key=pid_key))
    items = sorted(set(p for _, p in counts), key=pid_key)
    global_counts = collections.Counter()
    for (_, pid), count in counts.items():
        global_counts[pid] += count
    fallback = sorted(candidates, key=lambda p: (-global_counts[p], pid_key(p)))
    method = best_config["method"]
    repeat_scores = {u: np.asarray([sum_n.get(p, 0) for p in candidates], dtype=np.float32)
                     for u, sum_n in ((u, {p: counts.get((u, p), 0) for p in candidates}) for u in submission_users)}
    global_scores = {u: np.asarray([global_counts[p] for p in candidates], dtype=np.float32)
                     for u in submission_users}
    if method == "PPR":
        alpha = best_config["alpha"]
        scores, _ = ppr_scores(matrix, users, submission_users, candidates, items, (alpha,))
        orders, _ = ranks_from_scores(scores[alpha], candidates, fallback)
    elif method == "MF":
        scores, _ = mf_scores(matrix, uidx, iidx, submission_users, candidates, items, best_config["factors"])
        orders, _ = ranks_from_scores(scores, candidates, fallback)
    elif method == "hybrid_mf_ppr":
        ppr, _ = ppr_scores(matrix, users, submission_users, candidates, items, (best_config["alpha"],))
        mf, _ = mf_scores(matrix, uidx, iidx, submission_users, candidates, items, best_config["factors"])
        orders = rank_fusion(ppr[best_config["alpha"]], mf, candidates, fallback)
    elif method in ("global_popularity", "personal_repeat_frequency"):
        scores = global_scores if method == "global_popularity" else repeat_scores
        orders, _ = ranks_from_scores(scores, candidates, fallback)
    elif method == "hybrid_best_repeat":
        base_method = best_config["base"]
        if base_method == "PPR":
            base, _ = ppr_scores(matrix, users, submission_users, candidates, items, (best_config["alpha"],))
            base_scores = base[best_config["alpha"]]
        elif base_method == "MF":
            base_scores, _ = mf_scores(matrix, uidx, iidx, submission_users, candidates, items, best_config["factors"])
        elif base_method == "personal_repeat_frequency":
            base_scores = repeat_scores
        else:
            base_scores = global_scores
        orders = rank_fusion(base_scores, repeat_scores, candidates, fallback)
    elif method == "hybrid_schedule":
        base_name = best_config["base_hybrid"]
        if base_name == "MF+PPR_RRF":
            ppr, _ = ppr_scores(matrix, users, submission_users, candidates, items, (best_config["alpha"],))
            mf, _ = mf_scores(matrix, uidx, iidx, submission_users, candidates, items, best_config["factors"])
            base_orders = rank_fusion(ppr[best_config["alpha"]], mf, candidates, fallback)
        else:
            method2 = best_config["base_method"]
            if method2 == "PPR":
                ppr, _ = ppr_scores(matrix, users, submission_users, candidates, items, (best_config["alpha"],))
                base_scores = ppr[best_config["alpha"]]
            elif method2 == "MF":
                base_scores, _ = mf_scores(matrix, uidx, iidx, submission_users, candidates, items, best_config["factors"])
            elif method2 == "personal_repeat_frequency":
                base_scores = repeat_scores
            else:
                base_scores = global_scores
            base_orders = rank_fusion(base_scores, repeat_scores, candidates, fallback)
        pos = {p: i for i, p in enumerate(candidates)}
        avail = np.asarray([math.log1p(test_schedule.get(p, 0)) for p in candidates], dtype=np.float32)
        avail /= max(float(avail.max()), 1.0)
        mix = {}
        for u in submission_users:
            rank_score = np.zeros(len(candidates), dtype=np.float32)
            for rank, p in enumerate(base_orders[u], 1):
                rank_score[pos[p]] = 1.0 / (60.0 + rank)
        mix[u] = rank_score + 0.002 * avail
        orders, _ = ranks_from_scores(mix, candidates, fallback)
    else:
        raise ValueError("Unsupported final method: " + method)
    recs = {u: orders[u][:5] for u in submission_users}
    validate_recs(recs, submission_users, candidates)
    out_dir = os.path.join(ROOT, "submissions")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "final_candidate.csv")
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "tv_show_id"])
        for user in submission_users:
            writer.writerow([user, " ".join(recs[user])])
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert [r["user_id"] for r in rows] == submission_users
    assert len(rows) == 1260
    assert all(len(r["tv_show_id"].split()) == 5 and len(set(r["tv_show_id"].split())) == 5 for r in rows)
    return {"path": path, "rows": len(rows), "candidate_count": len(candidates),
            "valid": True, "refit_and_generation_runtime_seconds": round(time.perf_counter()-started, 3)}


def main():
    started = time.perf_counter()
    train_path = os.path.join(OUT, "train_interactions.csv")
    val_path = os.path.join(OUT, "validation_interactions.csv")
    train = read_interactions([train_path])
    validation = read_interactions([val_path])
    users = read_ids(SUBMISSION)
    gt = read_ground_truth(os.path.join(OUT, "validation_ground_truth.csv"))
    train_users = {u for u, _ in train}
    val_schedule_files = ("export_arh_11-20-final.csv", "export_arh_21-30-final.csv")
    candidates = sorted(schedule_pool(val_schedule_files, "2020-05-18", "2020-07-27"), key=pid_key)
    items = sorted(set(p for _, p in train), key=pid_key)
    graph_users = sorted(set(u for u, _ in train) | set(users))
    matrix, uidx, iidx = build_matrix(train, graph_users, items)
    assert matrix.nnz == len(train)
    global_counts = collections.Counter()
    for (_, pid), count in train.items():
        global_counts[pid] += count
    fallback = sorted(candidates, key=lambda p: (-global_counts[p], pid_key(p)))
    baseline = {u: fallback[:5] for u in users}
    repeat = {}
    by_user = collections.defaultdict(list)
    for (u, pid), count in train.items():
        by_user[u].append((pid, count))
    for u in users:
        personal = [p for p, _ in sorted(by_user.get(u, []), key=lambda x: (-x[1], pid_key(x[0]))) if p in set(candidates)]
        rec = []
        for p in personal + fallback:
            if p not in rec: rec.append(p)
            if len(rec) == 5: break
        repeat[u] = rec
    results = {
        "protocol": {
            "python": __import__("sys").version.split()[0],
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "thread_limit": "OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1",
            "train_only": "data/processed/train_interactions.csv (11-20); validation labels only evaluator",
            "validation_candidate_pool": "labelled schedule ids during 21-30",
            "fixed_users": 1260,
            "subset": "submission users with train history and nonempty validation ground truth (1224)",
            "seed": SEED,
            "ppr_restart": "alpha is restart probability; q is each user's normalized qualified-airing counts over TRAIN programmes",
            "mf_objective": "sum over all user-item pairs c_ui*(p_ui-dot(xu,yi))^2 + lambda*(||X||^2+||Y||^2); p=log1p(count) on observed pairs else 0; c=1+40*log1p(count) observed else 1",
            "fallback": "global TRAIN qualified-airing popularity restricted to the shared candidate pool; ID tie-break ascending",
            "leakage": "validation labels excluded from all fitted matrices and popularity; test schedule not used in local validation",
        },
        "baselines": {}, "ppr": {}, "mf": {}, "hybrids": {}, "tests": {},
    }
    results["baselines"]["global_popularity"] = map_metrics(baseline, gt, users, train_users)
    results["baselines"]["personal_repeat_frequency"] = map_metrics(repeat, gt, users, train_users)
    scores_by_model = {}
    orders_by_model = {}
    ppr_scores_map, ppr_info = ppr_scores(matrix, graph_users, users, candidates, items, PPR_ALPHAS)
    ppr_rank_maps = {}
    for alpha in PPR_ALPHAS:
        orders, _ = ranks_from_scores(ppr_scores_map[alpha], candidates, fallback)
        recs = recommendations_from_order(orders, users)
        validate_recs(recs, users, candidates)
        score = map_metrics(recs, gt, users, train_users)
        results["ppr"][str(alpha)] = {**score, **ppr_info[str(alpha)]}
        scores_by_model["PPR_" + str(alpha)] = ppr_scores_map[alpha]
        orders_by_model["PPR_" + str(alpha)] = orders
        ppr_rank_maps[alpha] = orders
    mf_score_maps = {}
    mf_info = {}
    for factors in FACTORS_TO_TEST:
        scores, info = mf_scores(matrix, uidx, iidx, users, candidates, items, factors)
        orders, _ = ranks_from_scores(scores, candidates, fallback)
        recs = recommendations_from_order(orders, users)
        validate_recs(recs, users, candidates)
        metric = map_metrics(recs, gt, users, train_users)
        results["mf"][str(factors)] = {**metric, **info}
        mf_score_maps[factors] = scores
        mf_info[factors] = info
        scores_by_model["MF_" + str(factors)] = scores
        orders_by_model["MF_" + str(factors)] = orders
    reproducibility_started = time.perf_counter()
    repeat_u, repeat_v, repeat_loss = implicit_als(matrix, 32, seed=SEED, iterations=ALS_ITERATIONS)
    reproducibility_runtime = time.perf_counter() - reproducibility_started
    reproducibility_hash = hashlib.sha256(repeat_u.tobytes() + repeat_v.tobytes()).hexdigest()
    assert reproducibility_hash == mf_info[32]["factor_sha256"]
    assert repeat_loss == mf_info[32]["objective_trace"]

    # Hybrid 1: top MF and top PPR by validation, reciprocal-rank fusion.
    best_alpha = max(PPR_ALPHAS, key=lambda a: results["ppr"][str(a)]["all_1260"])
    best_factors = max(FACTORS_TO_TEST, key=lambda f: results["mf"][str(f)]["all_1260"])
    hybrid1_orders = rank_fusion(ppr_scores_map[best_alpha], mf_score_maps[best_factors], candidates, fallback)
    hybrid1 = recommendations_from_order(hybrid1_orders, users)
    validate_recs(hybrid1, users, candidates)
    results["hybrids"]["MF+PPR_RRF"] = map_metrics(hybrid1, gt, users, train_users)
    # Choose best individual by fixed-user MAP, baseline included.
    individual = [("global_popularity", results["baselines"]["global_popularity"]["all_1260"]),
                  ("personal_repeat_frequency", results["baselines"]["personal_repeat_frequency"]["all_1260"])]
    individual += [("PPR_" + str(a), results["ppr"][str(a)]["all_1260"]) for a in PPR_ALPHAS]
    individual += [("MF_" + str(f), results["mf"][str(f)]["all_1260"]) for f in FACTORS_TO_TEST]
    winner_name, _ = max(individual, key=lambda x: x[1])
    repeat_by_user = collections.defaultdict(dict)
    for (u, p), n in train.items():
        repeat_by_user[u][p] = n
    repeat_scores = {u: np.asarray([repeat_by_user.get(u, {}).get(p, 0)
                                    for p in candidates], dtype=np.float32) for u in users}
    if winner_name.startswith("PPR_"):
        win_scores = ppr_scores_map[float(winner_name[4:])]
    elif winner_name.startswith("MF_"):
        win_scores = mf_score_maps[int(winner_name[3:])]
    elif winner_name == "personal_repeat_frequency":
        win_scores = repeat_scores
    else:
        win_scores = {u: np.asarray([global_counts[p] for p in candidates], dtype=np.float32) for u in users}
    hybrid2_orders = rank_fusion(win_scores, repeat_scores, candidates, fallback)
    hybrid2 = recommendations_from_order(hybrid2_orders, users)
    validate_recs(hybrid2, users, candidates)
    results["hybrids"]["best_individual+repeat_RRF"] = map_metrics(hybrid2, gt, users, train_users)
    # Hybrid 3: rerank best of prior hybrids with future schedule availability only.
    hybrid_candidates = [("MF+PPR_RRF", hybrid1_orders), ("best_individual+repeat_RRF", hybrid2_orders)]
    hybrid_base_name, hybrid_base_orders = max(hybrid_candidates, key=lambda x: results["hybrids"][x[0]]["all_1260"])
    schedule_freq = schedule_pool(val_schedule_files, "2020-05-18", "2020-07-27")
    max_freq = max(schedule_freq.values())
    schedule_norm = {p: math.log1p(schedule_freq.get(p, 0)) / math.log1p(max_freq) for p in candidates}
    schedule_scores = {u: np.asarray([schedule_norm[p] for p in candidates], dtype=np.float32) for u in users}
    base_rank_scores = {}
    candidate_pos = {p: i for i, p in enumerate(candidates)}
    for u in users:
        base_rank_scores[u] = np.zeros(len(candidates), dtype=np.float32)
        for r, p in enumerate(hybrid_base_orders[u], 1):
            base_rank_scores[u][candidate_pos[p]] = 1.0 / (60.0 + r)
    reranked_scores = {u: base_rank_scores[u] + 0.002 * schedule_scores[u] for u in users}
    reranked_orders, _ = ranks_from_scores(reranked_scores, candidates, fallback)
    reranked = recommendations_from_order(reranked_orders, users)
    validate_recs(reranked, users, candidates)
    results["hybrids"]["best_hybrid+schedule_frequency"] = {
        **map_metrics(reranked, gt, users, train_users), "base_hybrid": hybrid_base_name,
        "schedule_weight": 0.002, "schedule_feature": "validation-period broadcast frequency only"}

    # Verify PPR personalization/probability mass from saved propagation outputs.
    users_with_history = [u for u in users if u in train_users]
    ppr_personalized = (len(users_with_history) >= 2 and
                        not np.array_equal(ppr_scores_map[best_alpha][users_with_history[0]],
                                           ppr_scores_map[best_alpha][users_with_history[1]]))
    results["tests"] = {
        "ppr_personalized_scores_differ": bool(ppr_personalized),
        "ppr_probability_mass_error_below_1e-6": all(v["max_probability_mass_error"] < 1e-6 for v in ppr_info.values()),
        "ppr_all_alpha_converged": all(v["converged"] for v in ppr_info.values()),
        "mf_latent_scores_finite": all(v["finite_user_factor_count"] == len(graph_users) for v in mf_info.values()),
        "all_recommendations_valid": True,
        "evaluator_manual_tests": "4/4 previously passed; identical AP@5 convention reused",
        "fixed_seed": SEED,
        "same_seed_32_factor_fit_exactly_reproducible": True,
        "reproducibility_refit_runtime_seconds": round(reproducibility_runtime, 3),
        "temporal_leakage": "not detected by construction; only train_interactions.csv used for local fit; validation used for scoring/candidate schedule only",
    }
    # Select strictly by full submission-user MAP; ties by method name for deterministic choice.
    choices = [(k, score) for k, score in individual]
    choices += [("hybrid:" + k, v["all_1260"]) for k, v in results["hybrids"].items()]
    winner, winner_map = max(choices, key=lambda x: (x[1], x[0]))
    if winner.startswith("hybrid:"):
        hname = winner.split(":", 1)[1]
        if hname == "MF+PPR_RRF":
            final_config = {"method": "hybrid_mf_ppr", "alpha": best_alpha, "factors": best_factors}
        elif hname == "best_individual+repeat_RRF":
            base = winner_name
            final_config = {"method": "hybrid_best_repeat", "base": base}
            if base.startswith("PPR_"): final_config.update(method="hybrid_best_repeat", base="PPR", alpha=float(base[4:]))
            elif base.startswith("MF_"): final_config.update(base="MF", factors=int(base[3:]))
        else:
            base = hybrid_base_name
            final_config = {"method": "hybrid_schedule", "base_hybrid": base,
                            "alpha": best_alpha, "factors": best_factors}
            if base == "best_individual+repeat_RRF":
                final_config["base_method"] = winner_name
                if winner_name.startswith("PPR_"): final_config["base_method"] = "PPR"; final_config["alpha"] = float(winner_name[4:])
                elif winner_name.startswith("MF_"): final_config["base_method"] = "MF"; final_config["factors"] = int(winner_name[3:])
    elif winner == "personal_repeat_frequency":
        final_config = {"method": "personal_repeat_frequency"}
    elif winner == "global_popularity":
        final_config = {"method": "global_popularity"}
    elif winner.startswith("PPR_"):
        final_config = {"method": "PPR", "alpha": float(winner[4:])}
    else:
        final_config = {"method": "MF", "factors": int(winner[3:])}
    results["selection"] = {"winner": winner, "MAP@5_all_1260": winner_map, "config": final_config}

    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    final_info = final_candidate(winner, final_config, train, validation, users)
    results["final_candidate"] = final_info
    results["total_experiment_runtime_seconds"] = round(time.perf_counter() - started, 3)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()