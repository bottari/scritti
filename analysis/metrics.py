import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


DIMENSIONS = ("originality", "emotion", "imagery")


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"results.json not found at: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("results"), list):
        rows = data["results"]
    elif isinstance(data, list):
        rows = data
    else:
        raise ValueError("results.json must be a list or {'results': [...]}.")

    cleaned = []
    for row in rows:
        if not isinstance(row, dict):
            continue

        poem_id = row.get("poem_id", row.get("comparison_id", row.get("id")))
        model = row.get("model", "unknown-model")
        if poem_id is None:
            continue

        parsed = {"poem_id": str(poem_id), "model": str(model)}
        valid = True
        for dim in DIMENSIONS:
            try:
                score = float(row.get(dim))
            except (TypeError, ValueError):
                valid = False
                break
            if score < 1 or score > 5:
                valid = False
                break
            parsed[dim] = score

        if valid:
            cleaned.append(parsed)

    if not cleaned:
        raise ValueError("No valid rows with poem_id/comparison_id/id and 1-5 scores were found.")

    return cleaned


def avg_overall(row: dict) -> float:
    return (row["originality"] + row["emotion"] + row["imagery"]) / 3


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute poem review metrics from results.json")
    parser.add_argument("--input", default="results.json", help="Path to results.json")
    args = parser.parse_args()

    rows = load_results(Path(args.input))

    print(f"Average originality: {mean(r['originality'] for r in rows):.2f}")
    print(f"Average emotion: {mean(r['emotion'] for r in rows):.2f}")
    print(f"Average imagery: {mean(r['imagery'] for r in rows):.2f}")
    print()

    by_model = defaultdict(list)
    for row in rows:
        by_model[row["model"]].append(row)

    print("By model averages:")
    for model in sorted(by_model.keys()):
        model_rows = by_model[model]
        print(
            f"- {model}: overall {mean(avg_overall(r) for r in model_rows):.2f}, "
            f"originality {mean(r['originality'] for r in model_rows):.2f}, "
            f"emotion {mean(r['emotion'] for r in model_rows):.2f}, "
            f"imagery {mean(r['imagery'] for r in model_rows):.2f}"
        )
    print()

    per_poem_model = defaultdict(list)
    for row in rows:
        key = (row["poem_id"], row["model"])
        per_poem_model[key].append(avg_overall(row))

    print("Per-poem scores:")

    def poem_sort_key(value: str):
        return (0, int(value)) if value.isdigit() else (1, value)

    poem_ids = sorted({k[0] for k in per_poem_model.keys()}, key=poem_sort_key)
    for poem_id in poem_ids:
        print(f"Poem ID {poem_id}:")
        model_keys = sorted([k for k in per_poem_model.keys() if k[0] == poem_id], key=lambda x: x[1])
        for _, model in model_keys:
            poem_avg = mean(per_poem_model[(poem_id, model)])
            print(f"  {model}: avg score {poem_avg:.1f}")


if __name__ == "__main__":
    main()
