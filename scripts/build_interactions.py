#!/usr/bin/env python3
"""Build leakage-safe view/program interactions and temporal baselines."""
import bisect
import collections
import csv
import datetime as dt
import json
import os
import sqlite3

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(ROOT, "data", "processed")
os.makedirs(OUT, exist_ok=True)
DB_PATH = os.path.join(OUT, "overlap_segments.sqlite")
TRAIN_START = dt.datetime(2020, 3, 8, 0, 0, 0)
TRAIN_END = dt.datetime(2020, 5, 18, 0, 0, 0)
VALID_END = dt.datetime(2020, 7, 27, 0, 0, 0)
THRESHOLD = 0.80
SCHEDULE_FILES = [
    "export_arh_11-20-final.csv",
    "export_arh_21-30-final.csv",
]
SUBMISSION = os.path.join(RAW, "submission.csv")


def to_seconds(value, fmt):
    stamp = dt.datetime.strptime(value, fmt)
    return stamp.toordinal() * 86400 + stamp.hour * 3600 + stamp.minute * 60 + stamp.second


def from_seconds(value):
    day, sec = divmod(value, 86400)
    return dt.datetime.fromordinal(day) + dt.timedelta(seconds=sec)


def program_sort(program_id):
    try:
        return (0, int(program_id), program_id)
    except ValueError:
        return (1, program_id, program_id)


def load_schedule():
    # Canonical event identity deduplicates records repeated across schedule files.
    events = {}
    raw_labelled_rows = 0
    for name in SCHEDULE_FILES:
        path = os.path.join(RAW, name)
        with open(path, newline="", encoding="utf-8-sig") as source:
            for row in csv.DictReader(source):
                duration = int(row["duration"])
                if duration <= 0 or row["tv_show_id"] == "0":
                    continue
                raw_labelled_rows += 1
                start = to_seconds(row["start_time"], "%d.%m.%Y %H:%M:%S")
                event_key = (row["channel_id"], start, duration, row["tv_show_id"])
                events[event_key] = (row["channel_id"], start, start + duration,
                                     duration, row["tv_show_id"])
    by_channel = collections.defaultdict(list)
    for key, event in events.items():
        by_channel[event[0]].append((event[1], event[2], event[3], event[4], key))

    # Ambiguity: two distinct labelled programmes scheduled on one channel at
    # overlapping times. Exclude both whole event records and report the count.
    ambiguous = set()
    for channel, rows in by_channel.items():
        rows.sort()
        for i, left in enumerate(rows):
            j = i + 1
            while j < len(rows) and rows[j][0] < left[1]:
                right = rows[j]
                if right[4] != left[4] and right[3] != left[3]:
                    ambiguous.add(left[4])
                    ambiguous.add(right[4])
                j += 1
    for channel in by_channel:
        by_channel[channel] = [row for row in sorted(by_channel[channel])
                               if row[4] not in ambiguous]
    return by_channel, events, ambiguous, raw_labelled_rows


def overlap(a, b, c, d):
    return max(0, min(b, d) - max(a, c))


def merged_coverage(segments):
    """Return union seconds and longest continuous run; shared seconds count once."""
    total = longest = 0
    run_start = run_end = None
    for start, end in sorted(segments):
        if run_end is None or start > run_end:
            if run_end is not None:
                longest = max(longest, run_end - run_start)
            run_start, run_end = start, end
            total += end - start
        elif end > run_end:
            total += end - run_end
            run_end = end
    if run_end is not None:
        longest = max(longest, run_end - run_start)
    return total, longest


def make_segment_db():
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)
    con = sqlite3.connect(DB_PATH)
    con.execute("PRAGMA journal_mode=OFF")
    con.execute("PRAGMA synchronous=OFF")
    con.execute("PRAGMA temp_store=FILE")
    con.execute("""CREATE TABLE segments (
        phase TEXT NOT NULL, user_id TEXT NOT NULL, event_key TEXT NOT NULL,
        program_id TEXT NOT NULL, duration INTEGER NOT NULL,
        seg_start INTEGER NOT NULL, seg_end INTEGER NOT NULL,
        PRIMARY KEY (phase,user_id,event_key,seg_start,seg_end)
    ) WITHOUT ROWID""")
    by_channel, events, ambiguous, raw_labelled_rows = load_schedule()
    starts, prefix_ends = {}, {}
    for channel, rows in by_channel.items():
        starts[channel] = [r[0] for r in rows]
        running = 0
        prefix_ends[channel] = []
        for row in rows:
            running = max(running, row[1])
            prefix_ends[channel].append(running)

    t0 = to_seconds("2020-03-08 00:00:00", "%Y-%m-%d %H:%M:%S")
    split = to_seconds("2020-05-18 00:00:00", "%Y-%m-%d %H:%M:%S")
    v_end = to_seconds("2020-07-27 00:00:00", "%Y-%m-%d %H:%M:%S")
    train_candidate_ids = set()
    val_candidate_ids = set()
    for _, start, end, _, pid in events.values():
        if overlap(start, end, t0, split):
            train_candidate_ids.add(pid)
        if overlap(start, end, split, v_end):
            val_candidate_ids.add(pid)

    pending = []
    view_rows = 0
    with open(os.path.join(RAW, "dataset11-30.csv"), newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            view_rows += 1
            channel = row["vsetv_id"]
            if channel not in by_channel:
                continue
            vs = to_seconds(row["start_time"], "%Y-%m-%d %H:%M:%S")
            ve = to_seconds(row["stop_time"], "%Y-%m-%d %H:%M:%S")
            if ve <= vs:
                continue
            intervals = by_channel[channel]
            # Earliest interval that can overlap, using prefix maximum end.
            i = bisect.bisect_right(prefix_ends[channel], vs)
            while i < len(intervals) and intervals[i][0] < ve:
                ps, pe, duration, pid, key = intervals[i]
                lo, hi = max(vs, ps), min(ve, pe)
                if lo < hi:
                    for phase, bound_start, bound_end in (
                        ("train", t0, split), ("validation", split, v_end)):
                        a, b = max(lo, bound_start), min(hi, bound_end)
                        if a < b:
                            pending.append((phase, row["user_id"], repr(key), pid,
                                            duration, a, b))
                            if len(pending) >= 50000:
                                con.executemany("INSERT OR IGNORE INTO segments VALUES (?,?,?,?,?,?,?)", pending)
                                con.commit()
                                pending.clear()
                i += 1
    if pending:
        con.executemany("INSERT OR IGNORE INTO segments VALUES (?,?,?,?,?,?,?)", pending)
        con.commit()
    con.execute("CREATE INDEX seg_order ON segments(phase,user_id,event_key,seg_start,seg_end)")
    con.commit()
    metadata = {
        "input_view_rows_scanned": view_rows,
        "unique_labelled_schedule_events": len(events),
        "valid_labelled_schedule_rows_read": raw_labelled_rows,
        "duplicate_valid_labelled_schedule_rows_removed": raw_labelled_rows - len(events),
        "ambiguous_event_records_excluded": len(ambiguous),
        "train_candidate_programmes": len(train_candidate_ids),
        "validation_candidate_programmes": len(val_candidate_ids),
    }
    return con, events, ambiguous, train_candidate_ids, val_candidate_ids, metadata


def aggregate_events(con, events):
    """Merge overlapping/touching segments per user-airing; retain sum and max run."""
    event_duration = {repr(key): row[3] for key, row in events.items()}
    cursor = con.execute("""SELECT phase,user_id,event_key,program_id,duration,seg_start,seg_end
        FROM segments ORDER BY phase,user_id,event_key,seg_start,seg_end""")
    counts = {phase: {"sum": collections.Counter(), "continuous": collections.Counter()}
              for phase in ("train", "validation")}
    current = None
    segments = []

    def flush(key, pieces):
        if key is None:
            return
        phase, user, event_key, pid, duration = key
        total, longest = merged_coverage(pieces)
        if total / duration >= THRESHOLD:
            counts[phase]["sum"][(user, pid)] += 1
        if longest / duration >= THRESHOLD:
            counts[phase]["continuous"][(user, pid)] += 1

    for phase, user, event_key, pid, duration, start, end in cursor:
        key = (phase, user, event_key, pid, duration)
        if key != current:
            flush(current, segments)
            current = key
            segments = []
        segments.append((start, end))
    flush(current, segments)
    return counts


def write_interactions(path, counter):
    users = sorted({user for user, _ in counter})
    with open(path, "w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(["user_id", "tv_show_id", "qualified_airings", "binary_interaction"])
        for user, pid in sorted(counter, key=lambda x: (x[0], program_sort(x[1]))):
            writer.writerow([user, pid, counter[(user, pid)], 1])
    return users


def average_precision_at_5(ranked, relevant):
    """AP@5 denominator is min(5, |relevant|); empty relevance scores zero."""
    if not relevant:
        return 0.0
    hits = 0
    total = 0.0
    for rank, pid in enumerate(ranked[:5], 1):
        if pid in relevant:
            hits += 1
            total += hits / rank
    return total / min(5, len(relevant))


def evaluator_tests():
    tests = [
        (["a", "b", "c", "d", "e"], {"a", "b", "c", "d", "e"}, 1.0),
        (["x", "a", "b", "c", "d"], {"a", "b", "c", "d", "e"}, (1/2 + 2/3 + 3/4 + 4/5) / 5),
        (["a", "x", "b", "x", "x"], {"a", "b"}, (1 + 2/3) / 2),
        (["x"] * 5, set(), 0.0),
    ]
    for ranked, relevant, expected in tests:
        actual = average_precision_at_5(ranked, relevant)
        assert abs(actual - expected) < 1e-12, (ranked, relevant, expected, actual)
    assert merged_coverage([(0, 7), (4, 10)]) == (10, 10)
    assert merged_coverage([(0, 4), (6, 10)]) == (8, 4)


def ranked_recs(users, candidates, train_frequency, global_order):
    candidate_set = set(candidates)
    fallback = [pid for pid in global_order if pid in candidate_set]
    by_user = collections.defaultdict(list)
    for (user, pid), count in train_frequency.items():
        if pid in candidate_set:
            by_user[user].append((pid, count))
    results = {}
    for user in users:
        personalized = sorted(
            by_user.get(user, []),
            key=lambda x: (-x[1], program_sort(x[0])))
        chosen = []
        seen = set()
        for pid, _ in personalized:
            if pid not in seen:
                chosen.append(pid); seen.add(pid)
            if len(chosen) == 5:
                break
        for pid in fallback:
            if len(chosen) >= 5:
                break
            if pid not in seen:
                chosen.append(pid); seen.add(pid)
        if len(chosen) != 5:
            raise ValueError("Candidate pool cannot fill five unique ids")
        results[user] = chosen[:5]
    return results


def top_ground_truth(validation_counts):
    by_user = collections.defaultdict(list)
    for (user, pid), count in validation_counts.items():
        by_user[user].append((pid, count))
    return {user: [pid for pid, _ in sorted(items, key=lambda x: (-x[1], program_sort(x[0])))[:5]]
            for user, items in by_user.items()}


def evaluate(recommendations, ground_truth, users, train_users):
    all_scores = [average_precision_at_5(recommendations[u], ground_truth.get(u, set())) for u in users]
    eligible = [u for u in users if u in train_users and ground_truth.get(u)]
    eligible_scores = [average_precision_at_5(recommendations[u], ground_truth[u]) for u in eligible]
    return {
        "fixed_submission_users": len(users),
        "users_with_train_history_and_validation_ground_truth": len(eligible),
        "MAP@5_all_submission_users_empty_ground_truth_zero": sum(all_scores) / len(users),
        "MAP@5_users_with_train_history_and_validation_ground_truth":
            sum(eligible_scores) / len(eligible) if eligible else 0.0,
    }


def main():
    evaluator_tests()
    con, events, ambiguous, train_candidates, val_candidates, metadata = make_segment_db()
    counts = aggregate_events(con, events)
    con.close()
    tr = counts["train"]["sum"]
    va = counts["validation"]["sum"]
    train_binary = {key: 1 for key in tr}
    val_binary = {key: 1 for key in va}
    os.makedirs(OUT, exist_ok=True)
    write_interactions(os.path.join(OUT, "train_interactions.csv"), tr)
    write_interactions(os.path.join(OUT, "train_binary.csv"), train_binary)
    write_interactions(os.path.join(OUT, "validation_interactions.csv"), va)
    write_interactions(os.path.join(OUT, "validation_binary.csv"), val_binary)
    write_interactions(os.path.join(OUT, "train_continuous_interactions.csv"), counts["train"]["continuous"])
    write_interactions(os.path.join(OUT, "validation_continuous_interactions.csv"), counts["validation"]["continuous"])

    # Ground truth is the five most frequently qualified validation airings.
    gt_lists = top_ground_truth(va)
    gt = {user: set(items) for user, items in gt_lists.items()}
    with open(os.path.join(OUT, "validation_ground_truth.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f); writer.writerow(["user_id", "rank", "tv_show_id", "qualified_airings"])
        for user in sorted(gt_lists):
            for rank, pid in enumerate(gt_lists[user], 1):
                writer.writerow([user, rank, pid, va[(user, pid)]])

    submission_users = []
    with open(SUBMISSION, newline="", encoding="utf-8-sig") as f:
        submission_users = [row["user_id"] for row in csv.DictReader(f)]
    candidates = sorted(val_candidates, key=program_sort)
    global_counts = collections.Counter()
    for (_, pid), n in tr.items():
        if pid in val_candidates:
            global_counts[pid] += n
    global_order = sorted(candidates, key=lambda pid: (-global_counts[pid], program_sort(pid)))
    global_recs = {user: global_order[:5] for user in submission_users}
    repeat_recs = ranked_recs(submission_users, candidates, tr, global_order)
    global_metrics = evaluate(global_recs, gt, submission_users, {u for u, _ in tr})
    repeat_metrics = evaluate(repeat_recs, gt, submission_users, {u for u, _ in tr})

    result = {
        "config": {
            "train": "[2020-03-08 00:00:00, 2020-05-18 00:00:00)",
            "validation": "[2020-05-18 00:00:00, 2020-07-27 00:00:00)",
            "watched_rule": "union overlap coverage / full broadcast duration >= 0.80",
            "alternative": "longest merged continuous overlap / full broadcast duration >= 0.80",
            "ground_truth": "top five programs by qualified validation airing count; ties by numeric program id ascending",
            "AP@5": "sum precision at relevant ranks / min(5, number of relevant GT programs); empty GT contributes 0; all submission users included in overall MAP",
            "repeat_recommendations": "personal train airing-frequency rank, then identical global train-popularity fallback; repeats allowed",
            "candidate_pool": "labelled program ids scheduled in validation interval",
            "ambiguous_schedule": "exclude both labelled events if distinct programme ids overlap on same channel",
        },
        "data": metadata,
        "schedule_diagnostics": {
            "ambiguous_event_records_excluded": len(ambiguous),
            "unique_train_airing_event_records": len(events),
            "validation_candidate_programmes": len(candidates),
        },
        "interactions": {
            "train_sum_qualified_user_program_pairs": len(tr),
            "train_qualified_airings": sum(tr.values()),
            "train_binary_user_program_pairs": len(train_binary),
            "train_continuous_qualified_pairs": len(counts["train"]["continuous"]),
            "train_continuous_qualified_airings": sum(counts["train"]["continuous"].values()),
            "validation_sum_qualified_user_program_pairs": len(va),
            "validation_qualified_airings": sum(va.values()),
            "validation_binary_user_program_pairs": len(val_binary),
            "validation_continuous_qualified_pairs": len(counts["validation"]["continuous"]),
            "validation_continuous_qualified_airings": sum(counts["validation"]["continuous"].values()),
            "train_users": len({u for u, _ in tr}),
            "validation_users_with_ground_truth": len(gt),
            "validation_users_with_fewer_than_five_relevant_programmes": sum(len(x) < 5 for x in gt.values()),
        },
        "baselines": {
            "global_popularity": global_metrics,
            "personal_repeat_frequency": repeat_metrics,
        },
        "tests": {"evaluator_manual_examples": "4/4 passed", "candidate_count_at_least_five": len(candidates) >= 5},
    }
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    # Two fixed, inspectable user cases, selected from submission order.
    example_users = []
    for user in submission_users:
        if user in gt and user in {u for u, _ in tr}:
            example_users.append(user)
        if len(example_users) == 2:
            break
    with open(os.path.join(OUT, "personal_review_examples.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["user_id", "train_program_frequency_top5", "validation_ground_truth_top5", "global_popularity_top5", "repeat_frequency_top5"])
        for user in example_users:
            p = sorted(((pid, n) for (u, pid), n in tr.items() if u == user), key=lambda x: (-x[1], program_sort(x[0])))
            writer.writerow([user, " ".join(pid for pid, _ in p[:5]), " ".join(gt_lists[user]),
                             " ".join(global_recs[user]), " ".join(repeat_recs[user])])

    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("personal review users:", example_users)


if __name__ == "__main__":
    main()