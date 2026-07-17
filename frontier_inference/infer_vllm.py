"""Generate molecular geometries using a locally-hosted model via vLLM.

Bypasses the HTTP/API layer for max throughput on a GPU cluster — vLLM's
engine batches all prompts together and saturates GPUs much better than
serial async API calls.

Same prompts (prompts.py), same parsing (parse.py), same scoring
(bench_tools.py) — only the generation backend differs.

Usage:
    # Single SMILES (quick smoke test on a tiny Qwen model)
    python infer_vllm.py --smiles "CCO" --model Qwen/Qwen3-0.6B

    # Batch with Qwen3 thinking enabled
    python infer_vllm.py --input /path/to/test_set.json --output qwen.jsonl \\
        --model Qwen/Qwen3-0.6B --enable-thinking

    # Multi-GPU tensor-parallel (for models that don't fit on one GPU)
    python infer_vllm.py --input test_set.json --output llama.jsonl \\
        --model meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4

    # Multi-GPU data-parallel (faster when the model fits on one GPU)
    python infer_vllm.py --input test_set.json --output qwen.jsonl \\
        --model Qwen/Qwen3-14B --data-parallel-size 4 --enable-thinking
"""
import argparse
import json
import os
import subprocess
import sys
import tempfile

from prompts import (
    XYZ_SYSTEM_PROMPT,
    ZMAT_SYSTEM_PROMPT,
    GEOMLLAMA_FORMATS,
    make_geomllama_prompt,
    make_user_prompt,
)
from parse import parse_result
from bench_tools import evaluate_xyz, print_stats_from_jsonl

FORMATS = ["xyz", "zmat", *sorted(GEOMLLAMA_FORMATS)]

# The geomllama LoRAs were trained on a bare alpaca template and may never emit
# an EOS token; without this they ramble past the geometry until max_tokens.
GEOMLLAMA_STOP = ["### Instruction:", "### Input:"]


def _split_thinking(raw: str) -> tuple[str, str | None]:
    """Pull a <think>...</think> block (Qwen3, DeepSeek-R1 style) out of raw output."""
    if "<think>" in raw and "</think>" in raw:
        s = raw.index("<think>")
        e = raw.index("</think>")
        reasoning = raw[s + len("<think>"):e].strip()
        clean = (raw[:s] + raw[e + len("</think>"):]).strip()
        return clean, reasoning
    return raw, None


def generate_batch_vllm(
    smiles_list: list[str],
    model: str,
    fmt: str = "xyz",
    temperature: float = 0.0,
    max_tokens: int = 2048,
    seed: int = 42,
    enable_thinking: bool = False,
    model_name: str | None = None,
    **llm_kwargs,
) -> list[dict]:
    """Generate geometries for many SMILES in a single batched vLLM call.

    `model` is the load path; `model_name` is an optional display label for the
    eval tables, since local checkpoints all share a basename like "merged".
    """
    # Lazy import so the data-parallel parent process never has to load vLLM.
    from vllm import LLM, SamplingParams

    geomllama = fmt in GEOMLLAMA_FORMATS

    # geomllama checkpoints ship a generation_config.json with do_sample/
    # temperature/top_p from the Llama base model. vLLM would adopt those as
    # defaults and silently make the run stochastic, so ignore them and let
    # SamplingParams below decide.
    if geomllama:
        llm_kwargs.setdefault("generation_config", "vllm")

    llm = LLM(model=model, seed=seed, **llm_kwargs)
    tokenizer = llm.get_tokenizer()

    stop = None
    prompts = []
    if geomllama:
        # A completion model, not a chat model: no chat template, no special
        # tokens. Applying one would put it off-distribution.
        stop = GEOMLLAMA_STOP
        prompts = [make_geomllama_prompt(smi, fmt) for smi in smiles_list]
    else:
        system_prompt = XYZ_SYSTEM_PROMPT if fmt == "xyz" else ZMAT_SYSTEM_PROMPT
        chat_kwargs = {"enable_thinking": True} if enable_thinking else {}
        for smi in smiles_list:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": make_user_prompt(smi)},
            ]
            prompts.append(tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                **chat_kwargs,
            ))

    sampling = SamplingParams(temperature=temperature, max_tokens=max_tokens,
                              seed=seed, stop=stop)
    outputs = llm.generate(prompts, sampling)

    results = []
    for smi, out in zip(smiles_list, outputs):
        raw = out.outputs[0].text.strip()
        clean, reasoning = _split_thinking(raw)
        row = {
            "smiles": smi,
            "model": model,
            "format": fmt,
            "raw_response": clean,
            "reasoning": reasoning,
        }
        if model_name:
            row["model_name"] = model_name
        results.append(row)
    return results


def _run_data_parallel(args, entries: list[dict]) -> None:
    """Spawn N child workers (one per GPU group), each running the single-proc path,
    then merge their JSONL outputs and print aggregate stats.

    Worker i gets GPUs [i*tp, ..., (i+1)*tp - 1] via CUDA_VISIBLE_DEVICES.
    """
    n = args.data_parallel_size
    tp = args.tensor_parallel_size

    workdir = tempfile.mkdtemp(prefix="vllm_dp_")
    print(f"Data parallel: {n} workers x TP={tp} = {n * tp} GPUs total")
    print(f"Working directory: {workdir}\n")

    # Interleaved split keeps molecule-size distribution balanced across workers.
    chunks = [entries[i::n] for i in range(n)]
    chunk_paths = [os.path.join(workdir, f"part{i}_in.json") for i in range(n)]
    out_paths   = [os.path.join(workdir, f"part{i}_out.jsonl") for i in range(n)]
    log_paths   = [os.path.join(workdir, f"part{i}.log") for i in range(n)]

    for path, chunk in zip(chunk_paths, chunks):
        with open(path, "w") as f:
            json.dump(chunk, f)

    procs = []
    for i in range(n):
        gpu_ids = ",".join(str(g) for g in range(i * tp, (i + 1) * tp))
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--input", chunk_paths[i],
            "--output", out_paths[i],
            "--model", args.model,
            "--format", args.format,
            "--max-tokens", str(args.max_tokens),
            "--tensor-parallel-size", str(tp),
            "--data-parallel-size", "1",
        ]
        if args.enable_thinking:
            cmd.append("--enable-thinking")
        if args.model_name:
            cmd += ["--model-name", args.model_name]
        log_f = open(log_paths[i], "w")
        print(f"  Worker {i}: GPUs [{gpu_ids}], {len(chunks[i])} molecules -> {log_paths[i]}")
        procs.append((subprocess.Popen(cmd, env=env, stdout=log_f, stderr=subprocess.STDOUT), log_f))

    print(f"\nWaiting for {n} workers...")
    failed = []
    for i, (p, log_f) in enumerate(procs):
        rc = p.wait()
        log_f.close()
        status = "ok" if rc == 0 else f"FAILED (exit {rc})"
        print(f"  Worker {i}: {status}")
        if rc != 0:
            failed.append(i)

    if failed:
        for i in failed:
            print(f"\n--- last 30 lines of {log_paths[i]} ---")
            with open(log_paths[i]) as f:
                print("".join(f.readlines()[-30:]))
        sys.exit(1)

    with open(args.output, "w") as outf:
        for path in out_paths:
            with open(path) as f:
                outf.write(f.read())
    print(f"\nMerged {n} chunk outputs -> {args.output}")

    print_stats_from_jsonl(args.output)
    print(f"\nWorker logs and chunks preserved in {workdir}")
    print(f"Clean up with: rm -rf {workdir}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smiles", type=str, help="Single SMILES (quick test)")
    parser.add_argument("--input", type=str, help="JSON file (list of {smiles, coordinates, ...})")
    parser.add_argument("--output", type=str, default="results.jsonl")
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--model-name", type=str, default=None,
                        help="Display label for the eval tables. Use for local checkpoints, "
                             "whose paths all end in the same basename (e.g. 'merged').")
    parser.add_argument("--format", type=str, default="xyz", choices=FORMATS)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--enable-thinking", action="store_true",
                        help="Qwen3-style thinking mode (passes enable_thinking=True to chat template).")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1,
                        help="Run N workers in parallel, each on its own GPU group "
                             "(uses N * tensor-parallel-size GPUs total). Faster than TP "
                             "when the model fits on tensor-parallel-size GPUs.")
    parser.add_argument("-n", type=int, default=None, help="Only process first N molecules")
    args = parser.parse_args()

    llm_kwargs = {"tensor_parallel_size": args.tensor_parallel_size}

    if args.smiles:
        results = generate_batch_vllm(
            [args.smiles], model=args.model, fmt=args.format,
            max_tokens=args.max_tokens, enable_thinking=args.enable_thinking,
            model_name=args.model_name,
            **llm_kwargs,
        )
        r = parse_result(results[0])
        print(f"SMILES: {r['smiles']}")
        print(f"Model:  {r['model']}")
        print(f"Format: {r['format']}")
        if r.get("reasoning"):
            print(f"\n[Thinking]\n{r['reasoning']}\n")
        print()
        print(r["parsed"] or f"[PARSE FAILED] Raw response:\n{r['raw_response']}")
        return

    if not args.input:
        parser.print_help()
        return

    with open(args.input) as f:
        entries = json.load(f)
    if args.n:
        entries = entries[:args.n]
    print(f"Loaded {len(entries)} molecules from {args.input}")

    if args.data_parallel_size > 1:
        _run_data_parallel(args, entries)
        return

    smiles_list = [e["smiles"] for e in entries]
    results = generate_batch_vllm(
        smiles_list, model=args.model, fmt=args.format,
        max_tokens=args.max_tokens, enable_thinking=args.enable_thinking,
        model_name=args.model_name,
        **llm_kwargs,
    )

    stats = {"success": 0, "wrong_syntax": 0, "wrong_atoms": 0}
    rmsds = []
    with open(args.output, "w") as f:
        for entry, r in zip(entries, results):
            r = parse_result(r)
            r["ground_truth_coordinates"] = entry["coordinates"]
            r["filename"] = entry.get("filename", "")
            if r["parsed"]:
                rmsd = evaluate_xyz(entry["coordinates"], r["parsed"], fmt=args.format)
                r["rmsd"] = rmsd
                if isinstance(rmsd, float):
                    stats["success"] += 1
                    rmsds.append(rmsd)
                elif rmsd == "Wrong syntax":
                    stats["wrong_syntax"] += 1
                elif rmsd == "Wrong number of atoms":
                    stats["wrong_atoms"] += 1
            else:
                r["rmsd"] = "Wrong syntax"
                stats["wrong_syntax"] += 1
            f.write(json.dumps(r) + "\n")

    total = sum(stats.values())
    print(f"\n{'='*50}")
    print(f"Results: {args.output}")
    print(f"Total: {total}")
    print(f"  Valid RMSD:          {stats['success']}")
    print(f"  Wrong syntax:        {stats['wrong_syntax']}")
    print(f"  Wrong atom count:    {stats['wrong_atoms']}")
    if rmsds:
        rmsds.sort()
        print(f"\nRMSD (Angstroms):")
        print(f"  Mean:   {sum(rmsds) / len(rmsds):.4f}")
        print(f"  Median: {rmsds[len(rmsds) // 2]:.4f}")
        print(f"  Min:    {min(rmsds):.4f}")
        print(f"  Max:    {max(rmsds):.4f}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
