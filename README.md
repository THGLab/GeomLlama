# GeomLlama

Code and data for (https://arxiv.org/abs/2607.13350)

**Models on HuggingFace:**
- [THGLab/Llama-3.1-8B-GeomLlama-zmatrix](https://huggingface.co/THGLab/Llama-3.1-8B-GeomLlama-zmatrix) (Z-matrix / Fenske-Hall format)
- [THGLab/Llama-3.1-8B-GeomLlama-xyz](https://huggingface.co/THGLab/Llama-3.1-8B-GeomLlama-xyz) (Cartesian XYZ format)

## Quickstart

Generate a 3D molecular geometry from a SMILES string:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "THGLab/Llama-3.1-8B-GeomLlama-zmatrix"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

prompt = (
    "### Instruction:\n"
    "You can generate accurate molecular coordinates from a prompt "
    "containing a SMILES string.\n\n"
    "### Input:\nSMILES: CC(=O)O\n\n"
    "### Response:\n"
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=1.0, top_p=0.95,
                         do_sample=True)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

For batch inference with vLLM (recommended for benchmarks):

```python
from geomllama.inference import InferenceEngine

engine = InferenceEngine("THGLab/Llama-3.1-8B-GeomLlama-zmatrix")
prompts = [...]  # list of formatted prompts
results = engine.generate(prompts, max_new_tokens=512, temperature=1.0, top_p=0.95)
```

## Installation

```bash
git clone https://github.com/THGLab/GeomLlama.git
cd GeomLlama
pip install -e .
```

That gives you the core library: geometry formats, the Z-matrix converter, and
evaluation. Most tasks need one of the two environments below.

## Environments

**Use two separate virtual environments.** The finetuning stack (axolotl) pins
`torch` and `transformers` versions that conflict with vLLM, so installing both
into one environment produces a broken install. This is why there is no `[all]`
extra — it would resolve to an environment that cannot run.

**1. Analysis** — data preparation, inference, evaluation, and figures. This is
the one you want for everything except training:

```bash
python -m venv .venv-analysis && source .venv-analysis/bin/activate
pip install -e ".[analysis]"
```

**2. Finetuning** — training only, in its own environment:

```bash
python -m venv .venv-training && source .venv-training/bin/activate
pip install -e ".[training]"
```

Then run each step under the matching environment: `scripts/prepare_*.py`,
`scripts/benchmark_*.py`, and `synth/` under *analysis*; `axolotl.cli.*` under
*finetuning*.

Narrower extras are available if you don't need the whole analysis stack:
`[data]` (prep scripts; includes torch-geometric, required to unpickle the GEOM
archives), `[synth]` (the Figure 3 pipeline), `[vllm]`, and `[dev]` (pytest).
`frontier_inference/` is self-contained and has its own `requirements.txt`.

**xTB** is not a pip dependency. The `synth/` optimization stage calls the
standalone `xtb` binary, so install it separately — `conda install -c
conda-forge xtb`, or a static build from
[grimme-lab/xtb](https://github.com/grimme-lab/xtb/releases). It is only needed
to regenerate Figure 3 from scratch; the cached results from `download_data.py
--results` do not require it. If `xtb` is not on your `PATH`, point to it:

```bash
export XTB_BIN=/path/to/xtb
```

### Reproducing the exact environment

The extras above use lower bounds, so a fresh install picks up current releases.
To match the environment that produced the paper's numbers instead, install from
the lock files:

```bash
# Analysis (Python 3.11.7)
pip install -r requirements-analysis.txt && pip install -e . --no-deps

# Finetuning (Python 3.11.6, separate venv)
pip install -r requirements-training.txt
```

Training ran on 4x A40 GPUs with CUDA 12.6.

### Known version issues

A few dependencies affect *results*, not just whether the code runs. If your
numbers drift from the paper, start here:

- **RDKit** (paper used 2025.9.6) — canonical SMILES generation and bond
  perception have changed across releases. Every prompt is keyed on a canonical
  SMILES, so a different RDKit can change model inputs and RMSD evaluation
  (`GetBestRMS`). This is the single most results-sensitive dependency.
- **scikit-learn** (1.8.0) — `prepare_qm9.py` defines the QM9 train/test split
  with `train_test_split(..., random_state=42)`. The split is only reproducible
  if the shuffling behavior matches. `tests/test_splits.py` verifies this by
  re-deriving the split and comparing filenames; run it after preparing data.
- **torch-geometric** (2.7.0) — required to unpickle the GEOM archives even
  though nothing imports it directly, since the pickles contain PyG `Data`
  objects. The raw pickles were written by an older PyG, so `prepare_*.py`
  reads `rdmol` via `data_obj.__dict__.get("rdmol")` rather than attribute
  access, which raises under current versions.
- **matplotlib / seaborn** (3.10.8 / 0.13.2) — figure rendering changes between
  releases. Figure 3 reproduces byte-identically only on the pinned versions;
  different versions still produce a correct figure, just not an identical file.
- **RDKit ≥ 2025** — the attribute-style `Chem.rdDetermineBonds` import no
  longer works. The code already uses the submodule form
  (`from rdkit.Chem.rdDetermineBonds import DetermineConnectivity,
  DetermineBondOrders`); this note is here in case you adapt that code.

## Downloading Data

```bash
# Download test/eval datasets (JSON splits + test dicts)
python download_data.py --datasets

# Download SFT training data (JSONL files)
python download_data.py --sft

# Download saved results (frontier inference + SmileyLlama)
python download_data.py --results

# Download the Alpaca instruction mix (needed only to train the hybrid models)
python download_data.py --alpaca

# Download everything
python download_data.py --all
```

Raw GEOM data is available from [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/JNGTDF). Raw QM9 data is available from [figshare](https://figshare.com/collections/Quantum_chemistry_structures_and_properties_of_134_kilo_molecules/978904).

## Data Provenance and Licensing

The **code** in this repository is licensed under [`LICENSE.txt`](LICENSE.txt)
(UC Berkeley OTL academic-use terms: educational, research, and not-for-profit
purposes). The **data archives** are hosted separately on figshare and carry
their own license, stated on each figshare entry.

### Provenance

The processed datasets and SFT training files retrieved by `download_data.py`
are derived from two publicly released datasets, each published by its authors
under a public-domain dedication:

| Source | Authors | Location | Upstream license |
|---|---|---|---|
| GEOM | Axelrod, Simon; Gomez-Bombarelli, Rafael | Harvard Dataverse, [doi:10.7910/DVN/JNGTDF](https://doi.org/10.7910/DVN/JNGTDF) | CC0 1.0 |
| QM9 ("Data for 133885 GDB-9 molecules") | Ramakrishnan, Raghunathan; Dral, Pavlo; Rupp, Matthias; von Lilienfeld, O. Anatole | figshare, [doi:10.6084/m9.figshare.978904_D12](https://doi.org/10.6084/m9.figshare.978904_D12) | CC0 |

Derived files in the figshare archives are released under CC0 1.0, matching
their upstream sources. The upstream terms above govern the underlying data, and
are reproduced here as stated by their publishers; consult the linked sources
directly for the authoritative terms.

If you use these datasets, please cite the original GEOM and QM9 papers in
addition to this work. CC0 does not legally require attribution, but citation
remains the scholarly norm.

### Alpaca instruction data

The instruction mix used by the hybrid training configs
(`configs/examples/geom_8b_hybrid_*.yml`) is **not** redistributed here.
`python download_data.py --alpaca` fetches it from
[tatsu-lab/stanford_alpaca](https://github.com/tatsu-lab/stanford_alpaca),
pinned to a specific commit and verified by checksum.

Alpaca is licensed CC BY-NC 4.0 and is derived from OpenAI `text-davinci-003`
output, so it is research-use-only and its terms are narrower than the rest of
the data here. It is fetched rather than bundled so that the figshare archives
contain only data derived for this paper and can carry their own license
cleanly.

## Reproducing Paper Results

### Table 1: QM9 Single-Geometry Benchmarks

**Using cached results** (no GPU needed):
```bash
cd frontier_inference
python eval_tables.py  # auto-discovers all *.jsonl result files, generates LaTeX table
```

**Running a new frontier model:**
```bash
cd frontier_inference
# Hosted API (requires API key in environment)
python run.py --input ../data/qm9/test_set.json --model gpt-4o --format zmat --output new_model.jsonl

# Local model via vLLM
python infer_vllm.py --input ../data/qm9/test_set.json --model <model_path> --format zmat --output new_model.jsonl

# Regenerate table (auto-discovers the new JSONL)
python eval_tables.py
```

**Fine-tuning a new QM9 model:**
```bash
# Step 1: Prepare data (if not using download_data.py)
python scripts/prepare_qm9.py --tarball <qm9_raw.tar.bz2> --output-dir data/qm9/ --convert-fh
python scripts/create_sft_data.py --input data/qm9/train_set.json --format ori_fh --output data/qm9/ori_fh.jsonl

# Step 2: Fine-tune with axolotl (or any Alpaca-compatible trainer)
python -m axolotl.cli.preprocess configs/examples/qm9_3b_ori_fh.yml
accelerate launch -m axolotl.cli.train configs/examples/qm9_3b_ori_fh.yml

# Step 3: Benchmark
python scripts/benchmark_qm9.py --model-path <merged_model> --test-set data/qm9/test_set.json --format ori_fh
```

### Tables 2 & 3: GEOM Conformer Ensemble Benchmarks

**Preparing data:**
```bash
# GEOM-QM9 (Table 2)
python scripts/prepare_geom.py --rdkit-folder <geom_rdkit_folder> --output-dir data/geom_qm9_large/ --convert-fh --save-test-dict
python scripts/create_sft_data.py --input data/geom_qm9_large/train_xyz.json --format ori_fh --fh-key fh_coordinates --output data/geom_qm9_large/train_fh/ori_fh.jsonl

# GEOM-Drugs-small (Table 3)
python scripts/prepare_drugs_small.py --raw-dir <drugs_raw> --output-dir data/geom_drugs_small/ --convert-fh --keep-hydrogens --save-test-dict
```

**Training** (hybrid model on QM9 + Drugs + Alpaca):

The hybrid configs mix in an instruction dataset at `data/alpaca/alpaca.jsonl`
to preserve language ability. Fetch it first with `python download_data.py
--alpaca`, which pulls the stock Alpaca set (52,002 records) from upstream and
writes it where the configs expect.

```bash
python -m axolotl.cli.preprocess configs/examples/geom_8b_hybrid_ori_fh.yml
accelerate launch -m axolotl.cli.train configs/examples/geom_8b_hybrid_ori_fh.yml
python -m axolotl.cli.merge_lora configs/examples/geom_8b_hybrid_ori_fh.yml --lora_model_dir outputs
```

**Inference + evaluation:**
```bash
# GEOM-QM9 (Table 2, threshold 0.5 A)
python scripts/benchmark_geom.py --model-path <merged_model> --test-dict data/geom_qm9_large/test_smiles_dict.pkl --format ori_fh --output results/geom_qm9.pkl

# GEOM-Drugs (Table 3, threshold 1.25 A)
python scripts/benchmark_geom_drugs.py --model-path <merged_model> --test-dict data/geom_drugs_small/test_smiles_dict.pkl --format ori_fh --output results/geom_drugs.pkl
```

**Temperature/top_p sweep:**
```bash
python scripts/sweep_geom_drugs.py --model-path <merged_model> --test-dict data/geom_drugs_small/test_smiles_dict.pkl --format ori_fh
```

Default sampling parameters: `temperature=1.0`, `top_p=0.95`.

### Figure 3: SmileyLlama Drug-like Molecules

**Rebuilding figures from cached data**:
```bash
python download_data.py --results  # downloads cached results + xTB pickles
python -m synth.plots              # generates figures/1k_smiles_ro4/combined_panel.png
SYNTH_DATASET=1k_smiles_ro5 python -m synth.plots  # Ro5 variant
python -m synth.property_compare   # generates figures/property_comparison_3way.png
```

See `synth/README.md` for the full 4-stage pipeline (SMILES generation, geometry generation, xTB optimization, plotting).

### Language Modeling Benchmarks

To verify that hybrid-trained models retain language ability (Table 4), run [lm-eval-harness](https://github.com/EleutherAI/lm-evaluation-harness):

```bash
lm_eval --model hf \
    --model_args pretrained=THGLab/Llama-3.1-8B-GeomLlama-zmatrix \
    --tasks sciq,arc_easy,boolq,lambada_openai,hellaswag,piqa,arc_challenge,winogrande,openbookqa,mmlu \
    --batch_size auto
```

## Training Configuration

All models use LoRA fine-tuning (r=32, alpha=16, dropout=0.05, target_linear=true)
with the AdamW 8-bit optimizer, a cosine LR schedule, bf16 precision, gradient
checkpointing, and flash attention. All runs train for 4 epochs.

The two model families differ as follows:

| | QM9 (Table 1) | GEOM hybrid (Tables 2+3) |
|---|---|---|
| Learning rate | 2e-4 | 3e-4 |
| Sequence length | 2048 | 4096 |
| Datasets | 1 (single-format JSONL) | 3 (GEOM-Drugs + GEOM-QM9 + Alpaca mix) |
| 8-bit loading | yes | no |

All runs used 4x A40 GPUs with `micro_batch_size: 1` and
`gradient_accumulation_steps: 1`, giving an effective batch size of 4. If you
train on a different number of GPUs, change `gradient_accumulation_steps` to keep
the effective batch at 4.

Llama configs pin `pad_token: <|finetune_right_pad_id|>` and override
`rope_scaling`; the Qwen configs do neither, since Qwen ships its own pad token.

See `configs/examples/` for ready-to-use axolotl configs. Any Alpaca-compatible trainer can use the JSONL files directly.

## Citation

```bibtex
@article{cavanagh2026geomllama,
      title={How Well Can Frontier Large Language Models Generate Structures? High Quality Prediction of Molecular Geometries with Help from Fine-Tuning}, 
      author={Joseph M. Cavanagh and Jonathan B. Arnold and Giovanni Battista Alteri and Andrew Gritsevskiy and Teresa Head-Gordon},
      year={2026},
      eprint={2607.13350},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2607.13350},
}
```
