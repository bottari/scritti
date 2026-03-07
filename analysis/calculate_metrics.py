import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


DIMENSIONS = ("originality", "emotion", "imagery")


def load_results(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Results file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict) and isinstance(data.get("results"), list):
        rows = data["results"]
    else:
        raise ValueError("results.json must be a list or {'results': [...]} structure.")

    cleaned = []
    for idx, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        poem = str(row.get("poem", "")).strip()
        reviewer = str(row.get("reviewer", f"reviewer_{idx}")).strip() or f"reviewer_{idx}"
        if not poem:
            continue

        scores = {}
        valid = True
        for dim in DIMENSIONS:
            value = row.get(dim)
            try:
                score = int(value)
            except (TypeError, ValueError):
                valid = False
                break
            if score < 1 or score > 5:
                valid = False
                break
            scores[dim] = score

        if valid:
            cleaned.append({"poem": poem, "reviewer": reviewer, **scores})

    if not cleaned:
        raise ValueError("No valid review rows found in results.json.")

    return cleaned


def cohen_kappa(pairs: list[tuple[int, int]], categories: tuple[int, ...] = (1, 2, 3, 4, 5)) -> float:
    n = len(pairs)
    if n == 0:
        return 0.0

    observed = sum(1 for a, b in pairs if a == b) / n

    count_a = {c: 0 for c in categories}
    count_b = {c: 0 for c in categories}
    for a, b in pairs:
        if a in count_a:
            count_a[a] += 1
        if b in count_b:
            count_b[b] += 1

    expected = 0.0
    for c in categories:
        p_a = count_a[c] / n
        p_b = count_b[c] / n
        expected += p_a * p_b

    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def summarize(rows: list[dict]) -> dict:
    averages = {}
    for dim in DIMENSIONS:
        averages[dim] = round(mean(r[dim] for r in rows), 3)
    averages["overall"] = round(mean(averages[d] for d in DIMENSIONS), 3)

    by_poem = defaultdict(list)
    for row in rows:
        by_poem[row["poem"]].append(row)

    poem_summaries = []
    for poem, poem_rows in by_poem.items():
        summary = {"poem": poem, "num_reviews": len(poem_rows)}
        for dim in DIMENSIONS:
            summary[f"{dim}_avg"] = round(mean(r[dim] for r in poem_rows), 3)
        poem_summaries.append(summary)
    poem_summaries.sort(key=lambda r: r["num_reviews"], reverse=True)

    agreement = {}
    for dim in DIMENSIONS:
        pairwise_kappas = []
        exact_match_rates = []

        poem_reviewer_scores = defaultdict(dict)
        for row in rows:
            poem_reviewer_scores[row["poem"]][row["reviewer"]] = row[dim]

        for r1, r2 in itertools.combinations(sorted({r["reviewer"] for r in rows}), 2):
            pairs = []
            for reviewer_scores in poem_reviewer_scores.values():
                if r1 in reviewer_scores and r2 in reviewer_scores:
                    pairs.append((reviewer_scores[r1], reviewer_scores[r2]))
            if pairs:
                pairwise_kappas.append(cohen_kappa(pairs))
                exact_match_rates.append(sum(1 for a, b in pairs if a == b) / len(pairs))

        agreement[dim] = {
            "avg_pairwise_kappa": round(mean(pairwise_kappas), 3) if pairwise_kappas else None,
            "avg_exact_match_rate": round(mean(exact_match_rates), 3) if exact_match_rates else None,
            "num_reviewer_pairs": len(pairwise_kappas),
        }

    return {
        "num_reviews": len(rows),
        "num_poems": len(by_poem),
        "averages": averages,
        "inter_rater_agreement": agreement,
        "per_poem_averages": poem_summaries,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Calculate average poem scores and inter-rater agreement from results.json."
    )
    parser.add_argument("--input", default="results.json", help="Path to results JSON file")
    parser.add_argument("--output", default=None, help="Optional output JSON path for summary")
    args = parser.parse_args()

    rows = load_results(Path(args.input))
    summary = summarize(rows)

    print(json.dumps(summary, indent=2, ensure_ascii=False))

    if args.output:
        out_path = Path(args.output)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nSaved summary to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
