#!/usr/bin/env python3
"""Streaming diagnosis of the Sweet.tv view-log / schedule join."""
import bisect
import collections
import csv
import datetime as dt
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data", "raw")
VIEW = os.path.join(RAW, "dataset11-30.csv")
SUBMISSION = os.path.join(RAW, "submission.csv")
SCHEDULES = [
    os.path.join(RAW, name)
    for name in (
        "export_arh_11-20-final.csv",
        "export_arh_21-30-final.csv",
        "export_arh_31-42-final.csv",
    )
]


def seconds(value, fmt):
    value = dt.datetime.strptime(value, fmt)
    return value.toordinal() * 86400 + value.hour * 3600 + value.minute * 60 + value.second


def show_time(value, fmt):
    return dt.datetime.strptime(value, fmt)


def format_seconds(value):
    day, second = divmod(value, 86400)
    return (dt.datetime.fromordinal(day).strftime("%Y-%m-%d") + " "
            + "%02d:%02d:%02d" % (second // 3600, (second % 3600) // 60, second % 60))


def inspect_file(path, date_column, date_format):
    months = collections.Counter()
    examples = []
    low = high = None
    count = 0
    with open(path, newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        header = reader.fieldnames
        for row in reader:
            stamp = show_time(row[date_column], date_format)
            low = stamp if low is None or stamp < low else low
            high = stamp if high is None or stamp > high else high
            months[stamp.strftime("%Y-%m")] += 1
            if len(examples) < 5:
                examples.append(row[date_column])
            count += 1
    return header, count, low, high, months, examples


def overlap_seconds(view_start, view_end, programme_start, programme_end):
    return max(0, min(view_end, programme_end) - max(view_start, programme_start))


def formula_examples():
    cases = [
        (0, 10, 0, 10, 10, "full"),
        (0, 10, 5, 15, 5, "partial"),
        (0, 10, 11, 20, 0, "disjoint"),
        (0, 10, 10, 20, 0, "touch right edge"),
        (10, 20, 0, 10, 0, "touch left edge"),
    ]
    print("FORMULA checks (seconds):")
    for vs, ve, ps, pe, expected, label in cases:
        actual = overlap_seconds(vs, ve, ps, pe)
        print("  %-16s expected=%d actual=%d %s" % (label, expected, actual, "OK" if actual == expected else "FAIL"))


def main():
    formula_examples()
    view_info = inspect_file(VIEW, "start_time", "%Y-%m-%d %H:%M:%S")
    print("\nTIME RANGES / distributions:")
    all_infos = [("views", view_info)]
    for path in SCHEDULES:
        info = inspect_file(path, "start_time", "%d.%m.%Y %H:%M:%S")
        all_infos.append((os.path.basename(path), info))
    for name, (_, count, low, high, months, examples) in all_infos:
        print("%s rows=%d min=%s max=%s" % (name, count, low, high))
        print("  year-month counts:", ", ".join("%s:%d" % x for x in sorted(months.items())))
        print("  timestamp examples:", examples)
    view_low, view_high = view_info[2], view_info[3]
    print("calendar date overlap with views:")
    for name, (_, _, low, high, _, _) in all_infos[1:]:
        print("  %s: %s" % (name, max(view_low.date(), low.date()) <= min(view_high.date(), high.date())))

    # Index every schedule interval by channel; include unlabelled rows.
    schedule = collections.defaultdict(list)
    schedule_date_channels = collections.defaultdict(set)
    schedule_channels = set()
    labeled_channels = set()
    schedule_rows = collections.Counter()
    bad_durations = []
    for path in SCHEDULES:
        with open(path, newline="", encoding="utf-8-sig") as source:
            for row in csv.DictReader(source):
                start = seconds(row["start_time"], "%d.%m.%Y %H:%M:%S")
                duration = int(row["duration"])
                channel = row["channel_id"]
                labelled = row["tv_show_id"] != "0"
                schedule_rows["all"] += 1
                schedule_rows["labelled"] += labelled
                schedule_rows["unlabelled"] += not labelled
                if duration <= 0:
                    bad_durations.append((os.path.basename(path), channel, row["start_time"], duration))
                schedule[channel].append((start, start + duration, labelled, row["tv_show_id"]))
                schedule_date_channels[(channel, row["start_time"][:10])] .add(labelled)
                schedule_channels.add(channel)
                if labelled:
                    labeled_channels.add(channel)
    starts = {}
    prefix_ends = {}
    for channel, intervals in schedule.items():
        intervals.sort()
        starts[channel] = [interval[0] for interval in intervals]
        max_end = 0
        ends = []
        for _, end, _, _ in intervals:
            max_end = max(max_end, end)
            ends.append(max_end)
        prefix_ends[channel] = ends

    # Test ids can be compared in constant time while streaming view rows.
    test_users = set()
    with open(SUBMISSION, newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            test_users.add(row["user_id"])

    view_channels = set()
    view_users = set()
    test_users_seen = set()
    test_users_on_common = set()
    common_channel_views = 0
    common_channel_users = set()
    same_channel_date_views = 0
    overlaps_all = 0
    overlaps_labelled = 0
    overlap_users_all = set()
    overlap_users_labelled = set()
    overlap_pairs_all = 0
    overlap_pairs_labelled = 0
    user_program_pairs = set()
    first_labelled_example = None
    view_rows = 0
    with open(VIEW, newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            view_rows += 1
            user = row["user_id"]
            channel = row["vsetv_id"]
            view_users.add(user)
            view_channels.add(channel)
            if user in test_users:
                test_users_seen.add(user)
            common = channel in schedule_channels
            if common:
                common_channel_views += 1
                common_channel_users.add(user)
                if user in test_users:
                    test_users_on_common.add(user)
                key = (channel, row["start_time"][:10].replace("-", "."))
                # Schedule day is DD.MM.YYYY; view day is YYYY-MM-DD.
                y, m, d = row["start_time"][:10].split("-")
                if (channel, "%s.%s.%s" % (d, m, y)) in schedule_date_channels:
                    same_channel_date_views += 1
            if not common:
                continue
            view_start = seconds(row["start_time"], "%Y-%m-%d %H:%M:%S")
            view_end = seconds(row["stop_time"], "%Y-%m-%d %H:%M:%S")
            intervals = schedule[channel]
            # Find the earliest interval that can still overlap: sorted starts
            # alone are insufficient because an earlier long interval may span
            # the view. Prefix maximum ends provide a safe lower bound.
            i = bisect.bisect_right(prefix_ends[channel], view_start)
            hit_all = hit_labelled = False
            while i < len(intervals) and intervals[i][0] < view_end:
                programme_start, programme_end, labelled, programme_id = intervals[i]
                amount = overlap_seconds(view_start, view_end, programme_start, programme_end)
                if amount > 0:
                    hit_all = True
                    overlap_pairs_all += 1
                    if labelled:
                        hit_labelled = True
                        overlap_pairs_labelled += 1
                        user_program_pairs.add((user, programme_id))
                        if first_labelled_example is None:
                            first_labelled_example = (
                                user, channel, row["start_time"], row["stop_time"],
                                programme_id, programme_start, programme_end, amount)
                i += 1
            if hit_all:
                overlaps_all += 1
                overlap_users_all.add(user)
            if hit_labelled:
                overlaps_labelled += 1
                overlap_users_labelled.add(user)

    common_channels = view_channels & schedule_channels
    print("\nJOIN (ALL %d view rows scanned):" % view_rows)
    print("view channels=%d schedule channels=%d common=%d" % (len(view_channels), len(schedule_channels), len(common_channels)))
    print("views on common channels=%d distinct users=%d" % (common_channel_views, len(common_channel_users)))
    print("common channels with any labelled schedule row=%d of %d" % (len(common_channels & labeled_channels), len(common_channels)))
    print("views sharing common-channel calendar date with a schedule row=%d" % same_channel_date_views)
    print("positive overlap: any airtime rows=%d pairs=%d users=%d" % (overlaps_all, overlap_pairs_all, len(overlap_users_all)))
    print("positive overlap: labelled rows=%d pairs=%d users=%d" % (overlaps_labelled, overlap_pairs_labelled, len(overlap_users_labelled)))
    print("unique positive-overlap user-programme pairs=%d" % len(user_program_pairs))
    if first_labelled_example:
        user, channel, vs, ve, program_id, ps, pe, amount = first_labelled_example
        print("first labelled overlap: user=%s channel/vsetv=%s view=[%s,%s] programme_id=%s programme=[%s,%s] overlap_seconds=%d" % (
            user, channel, vs, ve, program_id, format_seconds(ps), format_seconds(pe), amount))
    print("user log distinct=%d test ids=%d found in log=%d absent=%d" % (
        len(view_users), len(test_users), len(test_users & view_users), len(test_users - view_users)))
    print("test users with any views on common channels=%d; with labelled overlap=%d" % (
        len(test_users_on_common), len(test_users & overlap_users_labelled)))
    print("schedule rows=%d labelled=%d unlabelled=%d share_labelled=%.6f" % (
        schedule_rows["all"], schedule_rows["labelled"], schedule_rows["unlabelled"],
        schedule_rows["labelled"] / schedule_rows["all"]))
    print("nonpositive schedule durations=%d examples=%s" % (len(bad_durations), bad_durations[:5]))


if __name__ == "__main__":
    main()
