import json
from collections import defaultdict
from pathlib import Path

from flask import Flask, redirect, render_template, request, url_for

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
OUTPUTS_CANDIDATES = [
    BASE_DIR / "outputs.json",
    BASE_DIR / "poem_review_app_outputs" / "outputs.json",
]
RESULTS_PATH = BASE_DIR / "results.json"
SCORE_FIELDS = ("originality", "emotion", "imagery")


def parse_score(value: str, field_name: str) -> int:
    try:
        score = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} score must be an integer.") from exc

    if score < 1 or score > 5:
        raise ValueError(f"{field_name} score must be between 1 and 5.")

    return score


def _normalize_side_by_side(rows: list) -> list[dict]:
    cleaned = []
    for item in rows:
        if not isinstance(item, dict):
            continue

        pair_id = item.get("id")
        prompt = item.get("prompt")
        outputs = item.get("outputs")

        if pair_id is None or prompt is None or not isinstance(outputs, list):
            continue
        if len(outputs) < 2:
            continue

        normalized_outputs = []
        for out in outputs[:2]:
            if not isinstance(out, dict):
                continue
            model = out.get("model")
            poem = out.get("poem")
            if model is None or poem is None:
                continue
            normalized_outputs.append({"model": str(model), "poem": str(poem)})

        if len(normalized_outputs) == 2:
            cleaned.append(
                {
                    "id": str(pair_id),
                    "prompt": str(prompt),
                    "outputs": normalized_outputs,
                }
            )
    return cleaned


def _normalize_flat(rows: list) -> list[dict]:
    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for item in rows:
        if not isinstance(item, dict):
            continue
        poem_id = item.get("id")
        prompt = item.get("prompt")
        model = item.get("model")
        poem = item.get("poem")
        if poem_id is None or prompt is None or model is None or poem is None:
            continue
        grouped[(str(poem_id), str(prompt))].append({"model": str(model), "poem": str(poem)})

    cleaned = []
    for (poem_id, prompt), outputs in grouped.items():
        if len(outputs) >= 2:
            cleaned.append({"id": poem_id, "prompt": prompt, "outputs": outputs[:2]})

    return cleaned


def _load_rows(path: Path) -> list:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and isinstance(data.get("outputs"), list):
        return data["outputs"]
    if isinstance(data, list):
        return data
    return []


def load_outputs() -> list[dict]:
    errors = []

    for candidate in OUTPUTS_CANDIDATES:
        if not candidate.exists():
            continue

        try:
            rows = _load_rows(candidate)
        except json.JSONDecodeError as exc:
            errors.append(f"{candidate}: invalid JSON ({exc})")
            continue

        side_by_side = _normalize_side_by_side(rows)
        if side_by_side:
            print(f"Loaded outputs from: {candidate}")
            return side_by_side

        flat = _normalize_flat(rows)
        if flat:
            print(f"Loaded flat outputs and converted to side-by-side from: {candidate}")
            return flat

        errors.append(f"{candidate}: no valid output records")

    if not any(p.exists() for p in OUTPUTS_CANDIDATES):
        raise FileNotFoundError(
            f"Missing outputs.json. Checked: {', '.join(str(p) for p in OUTPUTS_CANDIDATES)}"
        )

    raise ValueError(
        "No valid side-by-side items found in outputs.json. "
        "Expected either side-by-side (id,prompt,outputs[2]) or flat (id,model,prompt,poem).\n"
        + "\n".join(errors)
    )


def load_results() -> list[dict]:
    if not RESULTS_PATH.exists():
        return []

    data = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("results.json must be a list of review objects.")
    return data


def save_results(results: list[dict]) -> None:
    RESULTS_PATH.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


def next_item_for_reviewer(items: list[dict], results: list[dict], reviewer: str):
    reviewed_ids = {
        str(row.get("comparison_id", row.get("poem_id")))
        for row in results
        if isinstance(row, dict) and str(row.get("reviewer", "")).strip().lower() == reviewer.lower()
    }

    for item in items:
        if item["id"] not in reviewed_ids:
            return item
    return None


def mean_score(values: list[int]) -> float:
    return sum(values) / len(values)


@app.route("/", methods=["GET", "POST"])
def review_poem():
    error = None

    try:
        items = load_outputs()
        results = load_results()
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        return render_template("index.html", setup_error=str(exc), reviewer="", item=None, error=None)

    if request.method == "POST":
        reviewer = request.form.get("reviewer", "").strip()
        comparison_id = request.form.get("comparison_id", "").strip()

        if not reviewer:
            error = "Reviewer name is required."
        else:
            item = next((p for p in items if p["id"] == comparison_id), None)
            if item is None:
                error = "Comparison item not found for this submission."
            else:
                score_rows = []
                try:
                    for idx, output in enumerate(item["outputs"], start=1):
                        row = {
                            "reviewer": reviewer,
                            "comparison_id": item["id"],
                            "poem_id": item["id"],
                            "prompt": item["prompt"],
                            "model": output["model"],
                            "poem": output["poem"],
                        }
                        for field in SCORE_FIELDS:
                            form_key = f"m{idx}_{field}"
                            row[field] = parse_score(request.form.get(form_key), f"Model {idx} {field.title()}")

                        row["overall"] = round(mean_score([row[f] for f in SCORE_FIELDS]), 2)
                        score_rows.append(row)
                except ValueError as exc:
                    error = str(exc)
                else:
                    results.extend(score_rows)
                    save_results(results)
                    return redirect(url_for("review_poem", reviewer=reviewer))

    reviewer = request.args.get("reviewer", "").strip()
    item = next_item_for_reviewer(items, results, reviewer) if reviewer else None

    return render_template(
        "index.html",
        setup_error=None,
        reviewer=reviewer,
        item=item,
        error=error,
        total_items=len(items),
    )


if __name__ == "__main__":
    app.run(debug=True)

