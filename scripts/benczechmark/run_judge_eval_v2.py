#!/usr/bin/env python3
"""
LLM-as-Judge eval v2: comparison with Gemma 4 31B-it run under the Ollama-style
chat template (no forced thinking marker), with `thought\\n` prefix stripped
from model outputs.

Limited to 20 doc_ids common to the Ollama-style Gemma smoke and the existing
Qwen full runs. Gemma 4 26B-A4B-it is excluded — we don't have a clean
Ollama-template eval for it (would need another ~1.5h compute).

Usage:
    python run_judge_eval_v2.py --concurrent 8 --out judge_results_v2.json
"""

import argparse
import concurrent.futures as cf
import glob
import json
import random
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev

REPO_ROOT = Path("/home/spark/lm-evaluation-harness-czech")
RESULTS_BASE = REPO_ROOT / "results"

# (display name, results subdir, inner subdir, optional strip regex)
MODELS = [
    ("Qwen3.6-27B (chat+think_off)",
        "mvp_Qwen__Qwen3_6-27B_sumr_chat", "Qwen__Qwen3.6-27B", None),
    ("Qwen3.6-35B-A3B (chat+think_off)",
        "mvp_Qwen__Qwen3_6-35B-A3B_sumr_chat", "Qwen__Qwen3.6-35B-A3B", None),
    ("gemma-4-31B-it (chat+ollama_template)",
        "mvp_google__gemma-4-31B-it_ollama_chat", "google__gemma-4-31B-it",
        r"^[\"']?thought\s*\n+"),
]

JUDGE_SYSTEM = """Jsi expert hodnotitel českých textových souhrnů. Dostaneš originální článek, referenční (zlatý) souhrn napsaný editorem, a tři model-generované souhrny označené písmeny A, B, C.

Pro KAŽDÝ ze 3 souhrnů ohodnoť na škále 1–5 následující čtyři dimenze:

1. **Faithfulness** (věrnost): Souhlasí souhrn s fakty v článku? Žádné halucinace? (5 = vše věrné, 1 = závažné halucinace)
2. **Coverage** (pokrytí): Zachycuje souhrn klíčové body článku? (5 = vše podstatné, 1 = vynechá hlavní body)
3. **Fluency** (plynulost): Plynulá, gramaticky správná, idiomatická čeština? (5 = jako od rodilého mluvčího, 1 = lámaná čeština)
4. **Conciseness** (úspornost): Vhodná délka, hustý obsah, žádná zbytečnost? (5 = ideálně husté, 1 = upovídané nebo příliš krátké)

Hodnotíš výhradně proti referenci (zlatému souhrnu) a fakta v článku. Pořadí A/B/C je náhodné a nemá nic společného se kvalitou.

Vrať POUZE JSON podle dodaného schema. Žádný další text, žádné vysvětlení."""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        letter: {
            "type": "object",
            "properties": {
                "faithfulness": {"type": "integer", "minimum": 1, "maximum": 5},
                "coverage":     {"type": "integer", "minimum": 1, "maximum": 5},
                "fluency":      {"type": "integer", "minimum": 1, "maximum": 5},
                "conciseness":  {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["faithfulness", "coverage", "fluency", "conciseness"],
            "additionalProperties": False,
        }
        for letter in "ABC"
    },
    "required": list("ABC"),
    "additionalProperties": False,
}


def find_samples_file(results_dir: str, inner: str) -> Path:
    pattern = str(RESULTS_BASE / results_dir / inner / "samples_benczechmark_summarization_0_*.jsonl")
    matches = glob.glob(pattern)
    if not matches:
        raise FileNotFoundError(f"No samples file matching {pattern}")
    return Path(sorted(matches)[-1])


def load_samples(path: Path, strip_re: str = None) -> dict:
    out = {}
    sre = re.compile(strip_re) if strip_re else None
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            doc_id = d.get("doc_id")
            doc = d.get("doc", {})
            target = d.get("target", "")
            resp = d.get("filtered_resps", [""])[0] if d.get("filtered_resps") else ""
            if not resp and d.get("resps"):
                resp = d["resps"][0][0]
            resp = resp.strip()
            if sre:
                resp = sre.sub("", resp).strip()
            out[doc_id] = {"doc": doc, "gold": target, "output": resp}
    return out


def build_user_prompt(article: str, gold: str, outputs_by_letter: dict) -> str:
    parts = ["ČLÁNEK:", article, "", "REFERENČNÍ SOUHRN (zlatý):", gold, ""]
    for letter in "ABC":
        parts.append(f"SOUHRN {letter}:")
        parts.append(outputs_by_letter[letter])
        parts.append("")
    return "\n".join(parts)


def call_judge(prompt: str, model: str = "opus", timeout: int = 120) -> dict:
    cmd = [
        "claude", "-p",
        "--model", model,
        "--system-prompt", JUDGE_SYSTEM,
        "--output-format", "json",
        "--json-schema", json.dumps(JUDGE_SCHEMA, ensure_ascii=False),
        "--effort", "low",
        prompt,
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if res.returncode != 0:
        raise RuntimeError(f"claude CLI rc={res.returncode}: {res.stderr[:500]}")
    wrapper = json.loads(res.stdout)
    so = wrapper.get("structured_output")
    if isinstance(so, dict) and so:
        return so
    body = wrapper.get("result", "")
    return json.loads(body)


def judge_one(doc_id, article, gold, model_outputs, seed):
    rng = random.Random(seed)
    letters = list("ABC")
    rng.shuffle(letters)
    letter_to_model = dict(zip(letters, model_outputs.keys()))
    outputs_by_letter = {letter: model_outputs[name] for letter, name in letter_to_model.items()}
    prompt = build_user_prompt(article, gold, outputs_by_letter)
    judgement = call_judge(prompt)
    return {letter_to_model[letter]: judgement[letter] for letter in "ABC"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrent", type=int, default=8)
    ap.add_argument("--out", default="judge_results_v2.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-article-chars", type=int, default=6000)
    args = ap.parse_args()

    by_model = {}
    for name, results_dir, inner, strip_re in MODELS:
        path = find_samples_file(results_dir, inner)
        print(f"  loading {name}: {path.name} (strip={strip_re!r})", file=sys.stderr)
        by_model[name] = load_samples(path, strip_re)

    common = set(by_model[MODELS[0][0]].keys())
    for name, *_ in MODELS[1:]:
        common &= set(by_model[name].keys())
    common = sorted(common)
    print(f"  {len(common)} doc_ids common across {len(MODELS)} models", file=sys.stderr)

    pick = common  # judge all common docs
    print(f"  judging {len(pick)} doc_ids with --concurrent={args.concurrent}", file=sys.stderr)

    def task(idx_doc):
        idx, doc_id = idx_doc
        ref = by_model[MODELS[0][0]][doc_id]
        article = (ref["doc"].get("text") or ref["doc"].get("article") or "")[: args.max_article_chars]
        gold = ref["gold"]
        outputs = {name: by_model[name][doc_id]["output"] for name, *_ in MODELS}
        try:
            return doc_id, judge_one(doc_id, article, gold, outputs, seed=args.seed + idx)
        except Exception as e:
            return doc_id, {"_error": str(e)}

    results = {}
    with cf.ThreadPoolExecutor(max_workers=args.concurrent) as ex:
        futures = {ex.submit(task, (i, d)): d for i, d in enumerate(pick)}
        done = 0
        for fut in cf.as_completed(futures):
            doc_id, scores = fut.result()
            results[doc_id] = scores
            done += 1
            tag = "ERR" if "_error" in scores else "OK"
            print(f"  [{done:3d}/{len(pick)}] doc {doc_id} {tag}", file=sys.stderr)
            with open(args.out, "w") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)

    DIMS = ["faithfulness", "coverage", "fluency", "conciseness"]
    agg = {name: {d: [] for d in DIMS} for name, *_ in MODELS}
    n_err = 0
    for doc_id, scores in results.items():
        if "_error" in scores:
            n_err += 1
            continue
        for name, dims in scores.items():
            for d in DIMS:
                agg[name][d].append(dims[d])

    print("\n=== Aggregate (mean ± stdev) ===\n")
    print(f"{'Model':<42} " + " ".join(f"{d:>14}" for d in DIMS) + f" {'N':>5} {'avg':>6}")
    for name, *_ in MODELS:
        cells = []
        all_means = []
        for d in DIMS:
            vals = agg[name][d]
            if vals:
                m = mean(vals)
                s = stdev(vals) if len(vals) > 1 else 0
                cells.append(f"{m:.2f}±{s:.2f}")
                all_means.append(m)
            else:
                cells.append("    -    ")
        n = len(agg[name][DIMS[0]])
        avg = mean(all_means) if all_means else float("nan")
        print(f"{name:<42} " + " ".join(f"{c:>14}" for c in cells) + f" {n:>5} {avg:>6.2f}")

    if n_err:
        print(f"\n{n_err} judge calls errored — see {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
