"""Generate molecular geometries from SMILES using a frontier LLM.

Supports OpenAI, Anthropic (Claude), and Google (Gemini) models via litellm.
Set the corresponding API key(s): OPENAI_API_KEY, ANTHROPIC_API_KEY,
GEMINI_API_KEY.

Usage:
    # Single SMILES (quick test)
    python run.py --smiles "CCO"

    # Run on the QM9 test set JSON
    python run.py --input data/qm9/test_set.json --output results.jsonl

    # Limit to first N molecules
    python run.py --input test_set.json --output results.jsonl -n 10

    # Change model or format
    python run.py --smiles "CCO" --model gpt-4o-mini --format zmat

    # Use Claude or Gemini
    python run.py --smiles "CCO" --model claude-opus-4-5-20251001
    python run.py --smiles "CCO" --model gemini/gemini-2.5-pro

    # Enable extended thinking / reasoning (o-series, Claude 4.x, Gemini 2.5)
    python run.py --smiles "CCO" --model o3 --reasoning-effort high
"""

import argparse
import asyncio
import json
import os
import random
import sys

from tqdm import tqdm

from infer import generate_geometry, generate_batch_stream
from parse import parse_result
from bench_tools import evaluate_xyz, print_stats_from_jsonl


async def run_single(
    smiles: str, model: str, fmt: str, reasoning_effort: str | None = None,
    thinking_budget: int | None = None,
) -> None:
    result = await generate_geometry(
        smiles, model=model, fmt=fmt, reasoning_effort=reasoning_effort,
        thinking_budget=thinking_budget,
    )
    result = parse_result(result)

    print(f"SMILES: {result['smiles']}")
    print(f"Model:  {result['model']}")
    print(f"Format: {result['format']}")
    if result.get("usage"):
        u = result["usage"]
        print(f"Tokens: {u['prompt_tokens']} in / {u['completion_tokens']} out")
    if result.get("cost") is not None:
        print(f"Cost:   ${result['cost']:.6f}  (multiply by N molecules to estimate batch cost)")
    print()
    if result["parsed"]:
        print(result["parsed"])
    else:
        print("[PARSE FAILED] Raw response:")
        print(result["raw_response"])


async def run_batch(
    entries: list[dict], model: str, fmt: str,
    output_path: str, max_concurrency: int,
    reasoning_effort: str | None = None,
    thinking_budget: int | None = None,
    requests_per_minute: float | None = None,
    append: bool = False,
    score: bool = True,
) -> None:
    smiles_list = [e["smiles"] for e in entries]
    stats = {"success": 0, "wrong_syntax": 0, "wrong_atoms": 0, "api_error": 0}
    rmsds = []
    successful_results = []
    total_cost = 0.0
    total_in = 0
    total_out = 0

    # Stream results as they complete so the JSONL is written incrementally
    # (survives a crash / Ctrl-C for --resume) and a live bar shows progress.
    with open(output_path, "a" if append else "w") as f, \
            tqdm(total=len(entries), desc=model, unit="mol") as bar:
        async for i, r in generate_batch_stream(
            smiles_list, model=model, fmt=fmt, max_concurrency=max_concurrency,
            reasoning_effort=reasoning_effort,
            thinking_budget=thinking_budget,
            requests_per_minute=requests_per_minute,
        ):
            entry = entries[i]
            bar.update(1)
            if isinstance(r, Exception):
                tqdm.write(f"ERROR ({entry['smiles']}): {r}", file=sys.stderr)
                stats["api_error"] += 1
                continue
            r = parse_result(r)
            r["ground_truth_coordinates"] = entry["coordinates"]
            r["filename"] = entry.get("filename", "")

            if r.get("cost") is not None:
                total_cost += r["cost"]
            if r.get("usage"):
                total_in += r["usage"].get("prompt_tokens") or 0
                total_out += r["usage"].get("completion_tokens") or 0

            # Evaluate against ground truth (skipped in inference-only mode).
            # evaluate_xyz handles z-matrices via fmt (converts to Cartesian first).
            if not score:
                stats["success"] += 1 if r["parsed"] else 0
                stats["wrong_syntax"] += 0 if r["parsed"] else 1
            elif r["parsed"]:
                rmsd = evaluate_xyz(entry["coordinates"], r["parsed"], fmt=fmt)
                r["rmsd"] = rmsd
                if isinstance(rmsd, float):
                    stats["success"] += 1
                    rmsds.append(rmsd)
                    successful_results.append(r)
                elif rmsd == "Wrong syntax":
                    stats["wrong_syntax"] += 1
                elif rmsd == "Wrong number of atoms":
                    stats["wrong_atoms"] += 1
            else:
                r["rmsd"] = "Wrong syntax"
                stats["wrong_syntax"] += 1

            f.write(json.dumps(r) + "\n")
            f.flush()

            bar.set_postfix(ok=stats["success"], err=stats["api_error"])

    total = sum(stats.values())
    print(f"\n{'='*50}")
    print(f"Results: {output_path}")
    print(f"Total: {total}")
    print(f"  Valid RMSD:          {stats['success']}")
    print(f"  Wrong syntax:        {stats['wrong_syntax']}")
    print(f"  Wrong atom count:    {stats['wrong_atoms']}")
    print(f"  API errors:          {stats['api_error']}")
    if rmsds:
        rmsds.sort()
        mean = sum(rmsds) / len(rmsds)
        median = rmsds[len(rmsds) // 2]
        print(f"\nRMSD (Angstroms):")
        print(f"  Mean:   {mean:.4f}")
        print(f"  Median: {median:.4f}")
        print(f"  Min:    {min(rmsds):.4f}")
        print(f"  Max:    {max(rmsds):.4f}")
    if total_cost > 0 or total_in > 0:
        print(f"\nTokens: {total_in:,} in / {total_out:,} out ({total_in + total_out:,} total)")
        if total_cost > 0 and total > 0:
            print(f"Cost:   ${total_cost:.4f}  (~${total_cost / total:.6f} per molecule)")
    if successful_results:
        sample = random.choice(successful_results)
        from bench_tools import coordinates_to_string
        print(f"\nRandom sample: {sample['smiles']} (RMSD={sample['rmsd']:.4f})")
        print(f"\n  Generated:")
        for line in sample["parsed"].splitlines():
            print(f"    {line}")
        print(f"\n  Ground truth:")
        for line in coordinates_to_string(sample["ground_truth_coordinates"]).splitlines():
            print(f"    {line}")
    print(f"{'='*50}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", type=str, help="Single SMILES to test")
    parser.add_argument("--input", type=str, help="JSON file (list of {smiles, coordinates, ...})")
    parser.add_argument("--output", type=str, default="results.jsonl")
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument("--format", type=str, default="xyz", choices=["xyz", "zmat"])
    parser.add_argument("--max-concurrency", type=int, default=10)
    parser.add_argument("--reasoning-effort", type=str, default=None,
                        choices=["none", "low", "medium", "high"],
                        help="Extended thinking level on supported models "
                             "(OpenAI o-series, Claude 4.x, Gemini 2.5). "
                             "'none' explicitly disables reasoning.")
    parser.add_argument("--thinking-budget", type=int, default=None,
                        help="Direct thinking-token budget (Anthropic, Gemini). "
                             "Overrides --reasoning-effort. Use 0 to disable thinking. "
                             "Useful when 'low'/'medium'/'high' don't behave differently "
                             "(e.g. Gemini's dynamic thinking can use ~0 tokens on easy prompts).")
    parser.add_argument("--rpm", type=float, default=None,
                        help="Throttle to N requests per minute (e.g. --rpm 10 for free Gemini tier).")
    parser.add_argument("--resume", action="store_true",
                        help="Skip SMILES already present in --output and append new results. "
                             "Useful for retrying after API errors (503s, rate limits) without "
                             "redoing successful molecules.")
    parser.add_argument("--no-score", action="store_true",
                        help="Inference only: collect raw model output without scoring against "
                             "ground truth. Use to defer scoring (e.g. score later with "
                             "rescore.py). Omits the 'rmsd' field.")
    parser.add_argument("--stats", type=str, default=None,
                        help="Print aggregate stats (success counts, RMSD distribution) from a "
                             "results JSONL and exit. No inference. E.g. --stats results.jsonl")
    parser.add_argument("-n", type=int, default=None, help="Only process first N molecules")
    args = parser.parse_args()

    reasoning_effort = None if args.reasoning_effort == "none" else args.reasoning_effort

    if args.stats:
        print_stats_from_jsonl(args.stats)
        return

    if args.smiles:
        asyncio.run(run_single(
            args.smiles, args.model, args.format,
            reasoning_effort=reasoning_effort,
            thinking_budget=args.thinking_budget,
        ))
    elif args.input:
        with open(args.input) as f:
            entries = json.load(f)
        if args.n:
            entries = entries[:args.n]
        print(f"Loaded {len(entries)} molecules from {args.input}")

        append = False
        if args.resume and os.path.exists(args.output):
            done = set()
            with open(args.output) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        done.add(json.loads(line)["smiles"])
                    except (json.JSONDecodeError, KeyError):
                        continue
            before = len(entries)
            entries = [e for e in entries if e["smiles"] not in done]
            print(f"Resume: {len(done)} already in {args.output}, "
                  f"{before - len(entries)} skipped, {len(entries)} remaining")
            append = True
            if not entries:
                print("Nothing to do.")
                return

        asyncio.run(run_batch(
            entries, args.model, args.format,
            args.output, args.max_concurrency,
            reasoning_effort=reasoning_effort,
            thinking_budget=args.thinking_budget,
            requests_per_minute=args.rpm,
            append=append,
            score=not args.no_score,
        ))
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
