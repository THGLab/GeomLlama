# `synth/` — SmileyLlama -> geometry -> xTB

Produces the drug-like-molecule figures of **arXiv:2607.13350v1**
*(Cavanagh, Arnold, Alteri, Gritsevskiy, Head-Gordon — "How Well Can Frontier Large Language
Models Generate Structures?")*.

| Paper figure | Command | Output |
|---|---|---|
| **Fig 3(b,c,d)** — Ro4 panel | `python -m synth.plots` | `figures/1k_smiles_ro4/combined_panel.png` |
| Ro5 counterpart | `SYNTH_DATASET=1k_smiles_ro5 python -m synth.plots` | `figures/1k_smiles_ro5/combined_panel.png` |
| **Supplementary S3** — property distributions | `python -m synth.property_compare` | `figures/property_comparison_3way.png` |

```bash
python -m synth.plots                                 # Fig 3, from cached results (seconds)
SYNTH_DATASET=1k_smiles_ro5 python -m synth.plots     # Ro5 panel
python -m synth.property_compare                      # Supp S3 (~2 min, no GPU)
```

Everything editable is in `synth/config.py`.

## The published numbers, and how to not break them

**The paper is published. These numbers are frozen.** The pipeline reproduces the published
figures **byte-for-byte**; if a change moves any of them, the change is wrong.

| | Ro4 (Fig 3) | Ro5 |
|---|---|---|
| z-matrix median RMSD (pre->post opt) | **0.5085 A** | 0.6419 A |
| rdkit median RMSD | 0.7152 A | 0.8664 A |
| z-matrix median energy drop | 0.0165 Ha | 0.0296 Ha |
| rdkit median energy drop | 0.0593 Ha | 0.0767 Ha |
| **z-matrix atom mismatch** | **84/1000 = 8.4%** | **101/1000 = 10.1%** |
| z-matrix no-xTB | 18 (1.8%) | 40 (4.0%) |

These match the paper: *"GeomLlama has ~8.4% validity failures"*, *"84 of the 1000
Rule-of-Four molecules fail to generate a valid Z-matrix ... due to a wrong atom count"*, and
*"atom mismatches are only 10%"* for Ro5.

Reproducing them depends on exactly two settings, both defaults:

- **`CONFORMER_SELECTION = 'best_effort'`** — the published rule. Anything else is a different,
  unpublished claim (`'first'` gives 19.9% for Ro4, not 8.4%). `plots.py` prints a warning if you
  drift off it, and writes the rule into `figures/<dataset>/provenance.json`.
- **`--source published`** — reads `cache/<dataset>/xtb_*.pkl`, the artifacts of record.

### What `best_effort` means

Per molecule, out of the `N_CONFORMERS_PER_SMILES = 10` generations:

1. the first conformer that is **atom-matched and clash-free**;
2. else the **last atom-matched** one (they all clash);
3. else the 10th — nothing matched, so this molecule counts as the atom mismatch.

So the mismatch bar means **the model never once built the right molecule in 10 tries**, not
"the first try was wrong". RDKit gets **one** conformer (ETKDGv3, seed 42) and is 0% mismatch
*by construction* — it builds coordinates from the molecular graph, so it cannot emit the wrong
atoms. Giving RDKit 10 conformers would change nothing: it satisfies rule 1 on conformer 0 in
1000/1000 cases.

## Cache vs. regenerate

Two tracks. They must never be confused for one another.

**The cache is the artifact of record.** The published molecules cannot be regenerated:
SmileyLlama and the geometry model both sample at `T=1.0` with **no seed anywhere** in the
sampling path (`top_p=1.0` for SMILES, hardcoded at the notebook's call site), and the model
revision is unpinned. Re-running inference yields a *different, statistically equivalent* 1000
molecules — never the paper's. So `cache/` + `results/` are the only things that reproduce the
figures, and regeneration writes elsewhere and is labelled a fresh sample.

**Sources are explicit and never substituted.** `--source` is `published` or `notebook`; a missing
source is an error. The old `auto` mode silently preferred our cache and fell back to the
notebook's pickle — which is how the published panels came to source `z-matrix` from this
pipeline and `rdkit` from the notebook, two provenances in one figure with nothing recording it.
Both now come from `cache/`, verified identical to the notebook's (median RMSD 0.7152, dE 0.0593,
identical input coordinates on 1000/1000 molecules).

## Stages

| Stage | Command | Cost |
|---|---|---|
| 1. geometry generation | `python -m synth.generate --method z-matrix` | GPU, ~1 h |
| 2. xTB | `python -m synth.xtb_stage --all` | CPU, minutes on 64 cores |
| 3. figures | `python -m synth.plots` | seconds |

Each stage caches; stage 3 is all you need to rebuild the paper's figures.

> **Cache invalidation is `os.path.exists` only.** Editing `CONFORMER_SELECTION`, `XTB_MAX_ITER`,
> `XTB_ACCURACY` or `CLASH_THRESHOLD` does **not** invalidate `cache/<dataset>/xtb_*.pkl`. Re-run
> with `--force` after any such edit, or you will plot a stale pickle under a new label.

## Caveats a reader will otherwise find on their own

- **The Ro4 set is prompt-conditioned, not Ro4-filtered.** Only the prompt changed between the two
  runs; the post-hoc `is_druglike()` check used the **Ro5** thresholds for *both*. **340/1000** Ro4
  molecules violate at least one prompted Ro4 property (71 over 400 Da, 128 over logP 4, 192 over
  4 HBA, 2 over 4 HBD) — measured with the descriptors the notebook's own filter used
  (`Descriptors.MolWt/MolLogP`, `Lipinski.NumHDonors/NumHAcceptors`). That count is sensitive to
  the descriptor choice: `rdMolDescriptors.CalcNumHBA` counts acceptors more liberally and gives
  386/1000. Quote the definition alongside the number.
  The prompt still shifts the distribution hard — median MW 395.4 -> 320.4, MW>400 drops 470 -> 71 —
  so this is a genuine soft-constraint *adherence* result, which is how the paper frames it
  ("prompted with", "better overlap"). It is not a compliance guarantee.
- **The Ro4 molecules live in a file named `ro5_smiles.json`.** The notebook hardcoded that filename
  for both runs, so the sets differ only by parent directory (`results/1k_smiles_ro4/` vs
  `results/1k_smiles_ro5/`). Not a typo. Renaming would break the published caches.
- Both sets are internally consistent with the filter that was actually applied: **1000/1000** Ro4
  and **1000/1000** Ro5 molecules pass the notebook's Ro5 `is_druglike()` check.
- The two SMILES sets share only **11 of 1000** molecules; they are genuinely distinct populations.

## Why the stages are separate processes

The original notebook hung, for two structural reasons — both fixed by running the heavy stages as
separate processes:

1. **Forking a process pool after CUDA is initialized.** xTB forks 56 workers and RDKit embedding
   forks more, but the kernel had already loaded vLLM. A forked child inherits a broken CUDA context
   and hangs; the parent then blocks forever in `as_completed`.
   -> `synth/chem.py` and `synth/xtb_stage.py` never import torch, and the pool uses **spawn**.
2. **vLLM does not reliably release GPU memory between in-process model loads.** `del engine;
   torch.cuda.empty_cache()` does not tear down its workers.
   -> Each model loads in a **fresh subprocess** that exits.

Corollary: `synth/plots.py` cannot hang — no GPU, no forking, `Agg` backend.

## Bugs found in the original notebook

- **xTB results were never cached.** It recomputed hours of optimization on every run and persisted
  only at the metrics step, to a fixed `full_results_<name>.pkl` — so re-running the notebook
  silently destroyed the previous xTB run. The single most expensive footgun in the pipeline.
- **A silent fall-through**: `by_smiles = method_bysmiles.get('template_fh', ...)` never matched
  (the methods are named `templated_z-matrix` / `z-matrix`), so it always defaulted to the first
  method by accident.
- The post-hoc `ok()` hardness filter ran *after* xTB and silently narrowed the plotted population
  (heavy<30, HBA<8, aromatic rings<4, FractionCSP3<0.6) while the figures implied the full set.
  **Removed** — it was never used for any published figure.
- The Ro4 notebook config had **both** models set to `steelblue`, so they'd have been
  indistinguishable in the overlay.

## Colors

`#4682B4` / `#D95F02`, validated with `synth/_validate_palette.py`: CIEDE2000 >= 46 under normal,
deuteranopia, protanopia and tritanopia; both >= 3:1 contrast on white. S3's three-way palette
(`#2196F3` / `#E91E63` / `#1B5E20`) is validated separately — the dark green separates by
**lightness**, because a mid-green collapses against blue under tritanopia (dE=1.6). Re-run the
validator if you add or change a series.
