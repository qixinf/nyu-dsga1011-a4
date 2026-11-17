"""Build or inspect an assignment Q6 dev review TSV using saved evaluation outputs."""
import argparse
import csv
import pickle
from collections import Counter
from pathlib import Path

FIELDS = ("dev_line", "review_group", "question", "gold_sql", "pred_sql",
          "execution_error", "missing_records", "extra_records", "manual_error_type")
GROUPS = ("execution_error", "wrong_records", "matching_records")


def read_lines(path):
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]


def read_records(path):
    with path.open("rb") as stream:
        return pickle.load(stream)


def build_rows(experiment):
    root = Path(__file__).resolve().parent
    questions = read_lines(root / "data/dev.nl")
    gold_sql = read_lines(root / "data/dev.sql")
    pred_sql = read_lines(root / f"results/t5_ft_{experiment}_dev.sql")
    gold, _ = read_records(root / "records/ground_truth_dev.pkl")
    pred, errors = read_records(root / f"records/t5_ft_{experiment}_dev.pkl")
    rows = []
    for line, values in enumerate(zip(questions, gold_sql, pred_sql, gold, pred, errors, strict=True), 1):
        question, reference, prediction, expected, actual, error = values
        expected, actual = set(expected), set(actual)
        missing, extra = len(expected - actual), len(actual - expected)
        group = "matching_records"
        if error:
            group = "execution_error"
        elif missing or extra:
            group = "wrong_records"
        rows.append(dict(zip(FIELDS, (line, group, question, reference, prediction,
                                      error, missing, extra, ""), strict=True)))
    return rows


def analyze(rows, group, limit):
    total = len(rows)
    counts = Counter(r["review_group"] for r in rows)
    for name in GROUPS:
        print(f"{name}: {counts[name]}/{total}")
    matches = sum(int(r["missing_records"]) == int(r["extra_records"]) == 0 for r in rows)
    print(f"Record EM (including empty-set matches): {matches}/{total}")
    for field in ("execution_error", "manual_error_type"):
        print(f"\n{field} (count / all rows):")
        counts = Counter(r[field].strip() for r in rows if r[field].strip())
        for label, count in counts.most_common():
            print(f"  {count}/{total}  {label}")

    selected = [r for r in rows if r["review_group"] == group]
    for r in selected[:limit]:
        print(f"\nDev {r['dev_line']} | missing {r['missing_records']} | extra {r['extra_records']}")
        for field in ("question", "gold_sql", "pred_sql", "execution_error", "manual_error_type"):
            if r[field]:
                print(f"{field}: {r[field]}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--review", type=Path, help="Read an existing review TSV, including manual labels.")
    source.add_argument("--experiment", default="experiment", help="Experiment name for saved dev SQL and records.")
    parser.add_argument("--output", type=Path, help="Create a new TSV from saved dev outputs.")
    parser.add_argument("--group", choices=GROUPS, default="wrong_records")
    parser.add_argument("--limit", type=int, default=5, help="Number of examples to print for the selected group.")
    args = parser.parse_args()
    if args.limit < 0 or (args.review and args.output):
        parser.error("Use a nonnegative limit and supply --output only when building a TSV.")
    if args.review:
        with args.review.open(encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream, delimiter="\t")
            missing = set(FIELDS) - set(reader.fieldnames or ())
            if missing:
                parser.error(f"Missing TSV columns: {', '.join(sorted(missing))}")
            rows = list(reader)
    else:
        rows = build_rows(args.experiment)
        if args.output:
            with args.output.open("x", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter="\t")
                writer.writeheader()
                writer.writerows(rows)
            print(f"Wrote {args.output}")
    analyze(rows, args.group, args.limit)


if __name__ == "__main__":
    main()
