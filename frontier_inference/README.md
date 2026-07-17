# frontier_inference

Ask a frontier LLM to generate 3D molecular geometries (XYZ or Z-matrix)
directly from a SMILES string, using only its training knowledge — no
cheminformatics toolkits, no external calculators — then score the generated
geometry against a ground-truth structure via RMSD.

Supports **OpenAI**, **Anthropic (Claude)**, and **Google (Gemini)** models
through a single interface (`litellm`), plus extended **thinking/reasoning**
on models that support it. Local open-weight models run via vLLM.

## Setup

```bash
pip install -r requirements.txt   # litellm, asyncio-throttle, rdkit, numpy, scipy
```

Set the API key(s) for the provider(s) you want to use:

```bash
export OPENAI_API_KEY=...
export ANTHROPIC_API_KEY=...
export GEMINI_API_KEY=...
```

## Quick start

### Single SMILES

```bash
python run.py --smiles "CCO"
```

Prints the model's parsed geometry (or the raw response if parsing fails).

### Batch over a JSON dataset

The input JSON should be a list of `{"smiles": ..., "coordinates": [[elem, x, y, z], ...], "filename": ...}` entries, e.g. the QM9 test set:

```bash
python run.py --input /path/to/test_set.json --output results.jsonl
```

Results stream to `results.jsonl` **as each molecule completes** (one result dict
per line), with a live `tqdm` progress bar. Because writes are incremental, a
crash or Ctrl-C keeps everything finished so far — resume with `--resume`. At the
end it prints aggregate stats and a random successful sample.

```bash
# retry after rate limits / 503s without redoing finished molecules
python run.py --input test_set.json --output results.jsonl --resume

# free Gemini tier: throttle to 10 requests/min
python run.py --input test_set.json --output gemini.jsonl \
    --model gemini/gemini-2.5-flash --rpm 10

# first 10 molecules only
python run.py --input test_set.json --output results.jsonl -n 10
```

## Choosing a model

Pass any `litellm`-supported model name via `--model`:

| Provider  | Example `--model` values                            |
|-----------|-----------------------------------------------------|
| OpenAI    | `gpt-4o-mini`, `gpt-4o`, `o3`, `o4-mini`            |
| Anthropic | `claude-opus-4-5-20251001`, `claude-sonnet-4-5`     |
| Google    | `gemini/gemini-2.5-pro`, `gemini/gemini-2.5-flash`  |

```bash
python run.py --smiles "CCO" --model claude-opus-4-5-20251001
python run.py --smiles "CCO" --model gemini/gemini-2.5-pro
```

## Extended thinking / reasoning

Enable chain-of-thought reasoning on models that support it (OpenAI o-series, Claude 4.x, Gemini 2.5):

```bash
python run.py --smiles "CCO" --model o3                       --reasoning-effort high
python run.py --smiles "CCO" --model claude-opus-4-5-20251001 --reasoning-effort medium
python run.py --smiles "CCO" --model gemini/gemini-2.5-pro    --thinking-budget 2048
```

Non-reasoning models silently ignore the flag. `--thinking-budget` gives direct
token-level control (overrides `--reasoning-effort`; `0` disables thinking). The
thinking trace is captured in the result dict under `reasoning`.

## Output format

Geometry format is selected with `--format`:

- `xyz` (default): one `<element> <x> <y> <z>` line per atom, in Å.
- `zmat`: standard Z-matrix (internal coordinates), distances in Å, angles in degrees. Scored by converting to Cartesian first (see [converter.py](converter.py)).

```bash
python run.py --smiles "CCO" --format zmat
```

Each entry in `results.jsonl` has:

```jsonc
{
  "smiles": "...",
  "model":  "...",
  "format": "xyz" | "zmat",
  "raw_response": "...",        // verbatim from the LLM
  "reasoning": "..." | null,    // chain-of-thought, if the provider returned it
  "usage": {...}, "cost": 0.0,  // token usage / estimated cost, if available
  "parsed": "..." | null,       // cleaned geometry block, or null if parsing failed
  "rmsd": 0.1234 | "Wrong syntax" | "Wrong number of atoms",
  "ground_truth_coordinates": [...],
  "filename": "..."
  // after rescore_bruteforce.py, additionally:
  // "rmsd_graph": 0.12 | null, "rmsd_assignment": 0.10, "rmsd_method": "graph" | "assignment"
}
```

## Scoring & evaluation

Scoring is decoupled from inference. `run.py` scores inline by default, but you
can collect raw generations with `--no-score` and score later — useful while
iterating on the scoring pipeline, or for z-matrix runs.

### How RMSD is computed

RMSD needs an atom-to-atom correspondence between the model geometry and the
ground truth, accounting for molecular symmetry. Two methods, in
[bench_tools.py](bench_tools.py) and [rmsd_align.py](rmsd_align.py):

- **Graph RMSD** (`graph_rmsd`) — RDKit perceives bonds on both molecules
  (`DetermineBonds`) and `GetBestRMS` minimizes over graph automorphisms.
  Chemically strict, but returns `None` when a geometry is too implausible to
  bond-perceive.
- **Assignment RMSD** (`assignment_rmsd`) — a correspondence-free fallback: the
  minimum RMSD over all *element-preserving* atom permutations plus rigid
  alignment. It anchors the alignment by brute-forcing the (few) heavy-atom
  permutations, then solves the exact optimal atom assignment for a fixed
  alignment via the Hungarian algorithm (`scipy`) — hydrogens included — and
  refines with alternating Kabsch/Hungarian. This rescues structures RDKit can't
  bond-perceive. It is a *lower bound* on the graph RMSD (equal when the geometry
  is good), so treat it as a best-case number.

### Rescoring existing results (no inference)

```bash
# re-run the current evaluator over a results file, in place
python rescore.py results.jsonl [more.jsonl ...]

# correspondence-free rescoring in parallel: adds rmsd_graph / rmsd_assignment /
# rmsd_method, rescuing valid-atom structures RDKit couldn't bond-perceive
python rescore_bruteforce.py results.jsonl --jobs 8
```

`rescore_bruteforce.py` pins BLAS to one thread per worker for determinism and to
avoid core oversubscription under the process pool.

### Aggregate reports

```bash
# parse rate and atom-count validity for one or more files
python eval_stats.py results.jsonl [more.jsonl ...]

# two LaTeX-ready summary tables across runs (model x format):
#   Table 1 - most-complete RMSD (graph where possible, else assignment method)
#   Table 2 - strict RDKit-only RMSD + how often it can score
python eval_tables.py [file1.jsonl ...]          # LaTeX by default
python eval_tables.py --plain                    # human-readable

# quick stats from an already-scored file (no re-evaluation)
python run.py --stats results.jsonl
```

Every generation falls into exactly one of three buckets — has an RMSD, wrong
atom count, or invalid syntax — reported as nested percentages (syntax% of total,
atom% of parsable, graph% of valid-atom).

## CLI flags (`run.py`)

| Flag                  | Default         | Description                                         |
|-----------------------|-----------------|-----------------------------------------------------|
| `--smiles STR`        | —               | Single SMILES to run (mutually exclusive w/ input)  |
| `--input PATH`        | —               | JSON dataset of SMILES + ground-truth coordinates   |
| `--output PATH`       | `results.jsonl` | Where to write batch results (streamed incrementally)|
| `--model NAME`        | `gpt-4o-mini`   | Any litellm-supported model name                    |
| `--format {xyz,zmat}` | `xyz`           | Geometry format the LLM is asked to produce         |
| `--reasoning-effort`  | —               | `none`/`low`/`medium`/`high` — enables thinking     |
| `--thinking-budget N` | —               | Direct thinking-token budget (overrides the above)  |
| `--max-concurrency N` | `10`            | Parallelism for batch mode                          |
| `--rpm N`             | —               | Throttle to N requests/min (e.g. free Gemini tier)  |
| `--resume`            | off             | Skip SMILES already in `--output`, append new ones  |
| `--no-score`          | off             | Inference only — collect raw output, defer scoring  |
| `--stats PATH`        | —               | Print stats from a results JSONL and exit           |
| `-n N`                | —               | Only process the first N entries of the input       |

## Local open-weight models via vLLM

For Llama, Qwen, etc. on a GPU cluster, use [infer_vllm.py](infer_vllm.py) instead. It reuses the same prompts/parsing/scoring but generates via `vllm.LLM` directly — no HTTP layer, all prompts batched into a single engine call for max throughput.

```bash
pip install vllm

# Smoke test on a tiny Qwen3 model
python infer_vllm.py --smiles "CCO" --model Qwen/Qwen3-0.6B

# Qwen3 with thinking mode enabled
python infer_vllm.py --input test_set.json --output qwen.jsonl \
    --model Qwen/Qwen3-0.6B --enable-thinking

# Larger model split across 4 GPUs
python infer_vllm.py --input test_set.json --output llama.jsonl \
    --model meta-llama/Llama-3.1-70B-Instruct --tensor-parallel-size 4
```

`--enable-thinking` passes `enable_thinking=True` to the chat template (Qwen3-style). Output `<think>...</think>` blocks are stripped from `raw_response` and stored separately under `reasoning`.

## Files

- [run.py](run.py) — CLI entrypoint for hosted APIs (OpenAI / Claude / Gemini); streaming batch inference + inline scoring.
- [infer.py](infer.py) — async wrappers around `litellm.acompletion` (incl. `generate_batch_stream`).
- [infer_vllm.py](infer_vllm.py) — local open-weight inference via vLLM.
- [prompts.py](prompts.py) — system prompts for XYZ and Z-matrix modes.
- [parse.py](parse.py) — extract coordinate / Z-matrix blocks from LLM output.
- [bench_tools.py](bench_tools.py) — parsing, classification (`classify_geometry`), and graph-based RMSD (`evaluate_xyz`).
- [converter.py](converter.py) — Z-matrix ↔ Cartesian conversion.
- [rmsd_align.py](rmsd_align.py) — graph RMSD and correspondence-free assignment RMSD.
- [rescore.py](rescore.py) — re-score a results file with the current evaluator, in place.
- [rescore_bruteforce.py](rescore_bruteforce.py) — parallel correspondence-free rescoring.
- [eval_stats.py](eval_stats.py) — parse-rate and atom-count report.
- [eval_tables.py](eval_tables.py) — two LaTeX-ready summary tables across runs.
