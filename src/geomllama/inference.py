"""
LLM inference engine for molecular geometry generation.

Uses vllm for fast batch inference across available GPUs.
Supports data parallelism (multiple model replicas) for models
that fit on a single GPU.
"""

import multiprocessing as mp
import os
import pickle
import tempfile


def generate_feedback_fh(llm, prompt, n=1, temperature=0.6, top_p=0.95,
                         max_atoms=300, max_tokens_per_line=128):
    """Generate feedback_fh conformers with incremental XYZ injection.

    Generates Z-matrix lines one at a time, stopping at ' : ', computing the
    Cartesian coordinates of the newly placed atom, injecting them, and
    continuing. Batches all n conformers together at each step.

    Args:
        llm: vLLM LLM instance.
        prompt: Full inference prompt (from make_inference_prompt).
        n: Number of conformers to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        max_atoms: Safety limit on atoms per conformer.
        max_tokens_per_line: Max tokens generated per Z-matrix line.

    Returns:
        List of n generated output strings (Z-matrix with XYZ feedback).
    """
    from vllm import SamplingParams
    from geomllama.converter import fh_string_to_coordinates_raw

    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens_per_line,
        stop=[" : "],
    )

    active = list(range(n))
    full_texts = [""] * n
    zmat_lines_per = [[] for _ in range(n)]

    for _ in range(max_atoms):
        if not active:
            break

        batch_prompts = [prompt + full_texts[i] for i in active]
        outputs = llm.generate(batch_prompts, params)

        still_active = []
        for idx_in_batch, conf_idx in enumerate(active):
            new_text = outputs[idx_in_batch].outputs[0].text
            finish_reason = outputs[idx_in_batch].outputs[0].finish_reason

            if finish_reason != "stop":
                full_texts[conf_idx] += new_text
                continue

            zmat_line = new_text.strip()
            zmat_lines_per[conf_idx].append(zmat_line)

            zmat_str = '\n'.join(zmat_lines_per[conf_idx])
            try:
                coords = fh_string_to_coordinates_raw(zmat_str)
                last = coords[-1]
                xyz_str = f"{last[1]:.3f} {last[2]:.3f} {last[3]:.3f}"
            except Exception:
                full_texts[conf_idx] += new_text
                continue

            full_texts[conf_idx] += new_text + " : " + xyz_str + "\n"
            still_active.append(conf_idx)

        active = still_active

    return full_texts


def _parse_xyz_triple(text):
    """Extract the first three floats from `text` as (x, y, z).

    Returns None if fewer than three parseable floats are present.
    """
    toks = text.replace(",", " ").split()
    vals = []
    for t in toks:
        try:
            vals.append(float(t))
        except ValueError:
            continue
        if len(vals) == 3:
            return tuple(vals)
    return None


def generate_scaffolded_xyz(llm, prompt, labels, n=1, temperature=0.6,
                            top_p=0.95, max_tokens_per_atom=32,
                            fallback="0.000 0.000 0.000"):
    """Generate a labeled_xyz conformer atom-by-atom with a fixed scaffold.

    The atom labels (element + index, in canonical order) come from the SMILES
    graph and are emitted by the harness, NOT the model. For each label the
    harness appends ``"{label} "`` and the model fills only the ``"x y z"``
    coordinate (stop at newline). This guarantees exactly len(labels) atoms with
    the correct elements/order — 0% atom miscount by construction.

    Batches all n conformers together at each atom step.

    Args:
        llm: vLLM LLM instance.
        prompt: Full inference prompt (from make_inference_prompt).
        labels: List of "{Element}{index}" scaffold labels, in order.
        n: Number of conformers to generate.
        temperature: Sampling temperature.
        top_p: Nucleus sampling threshold.
        max_tokens_per_atom: Max tokens generated for one atom's coordinates.
        fallback: Coordinate string used if the model emits no parseable triple.

    Returns:
        List of n generated output strings (labeled_xyz, one atom per line).
    """
    from vllm import SamplingParams

    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens_per_atom,
        stop=["\n"],
    )

    full_texts = [""] * n
    for label in labels:
        batch_prompts = [prompt + full_texts[i] + label + " " for i in range(n)]
        outputs = llm.generate(batch_prompts, params)
        for i in range(n):
            new_text = outputs[i].outputs[0].text
            triple = _parse_xyz_triple(new_text)
            if triple is None:
                coord_str = fallback
            else:
                coord_str = f"{triple[0]:.3f} {triple[1]:.3f} {triple[2]:.3f}"
            full_texts[i] += f"{label} {coord_str}\n"

    return full_texts


def _parse_first_float(text):
    """First parseable float in `text`, or None."""
    for t in text.replace(",", " ").split():
        try:
            return float(t)
        except ValueError:
            continue
    return None


def generate_scaffolded_zmat(llm, prompt, scaffold, n=1, temperature=0.6,
                             top_p=0.95, max_tokens_per_value=16,
                             fallback=("1.000", "109.500", "180.000")):
    """Generate a template_fh conformer with a fixed scaffold.

    The atom labels AND the reference indices come from the SMILES graph and are
    emitted by the harness; the model fills only the internal-coordinate values
    (distance, angle, dihedral). This guarantees exactly len(scaffold) atoms with
    the correct elements/order/references.

    For atom i the harness emits ``"{label}"`` then, for each applicable field,
    ``" {ref} "`` followed by the model-generated value:
      i==0: ``"{label}"``                          (root)
      i==1: ``"{label} {r1} <dist>"``
      i==2: ``"{label} {r1} <dist> {r2} <ang>"``
      i>=3: ``"{label} {r1} <dist> {r2} <ang> {r3} <dih>"``
    All n conformers are batched together at each field step.

    Args:
        llm: vLLM LLM instance.
        prompt: Full inference prompt (from make_inference_prompt).
        scaffold: list of (label, (ref1, ref2, ref3)) with 0-based refs (None
                  where not applicable), from TemplateFH.scaffold_for(smiles).
        n: Number of conformers.
        temperature, top_p: sampling.
        max_tokens_per_value: token cap per internal-coordinate value.
        fallback: (dist, angle, dihedral) strings used if the model emits no
                  parseable number for a field.

    Returns:
        List of n template_fh output strings.
    """
    import re
    try:
        from vllm import SamplingParams
    except ImportError:
        from types import SimpleNamespace
        def SamplingParams(**kw):
            return SimpleNamespace(**kw)

    # Prefix ends at the ref number (no trailing space); model emits the
    # leading-space-+-value chunk as one BPE block. Training tokens at value
    # boundaries are ' -0.462' / ' 1.0782' (sign and leading space baked into
    # the same token); forcing the prefix to end past the space starts the
    # model at a token boundary it never saw in training, and the minus-sign
    # branch becomes unreachable — collapsing every signed dihedral to a
    # single mode. Stop only on newline; the leading space is part of the
    # value's BPE token, not a delimiter.
    params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens_per_value,
        stop=["\n"],
    )
    num_re = re.compile(r'\s*(-?\d+\.?\d*(?:[eE][+-]?\d+)?)')

    full_texts = [""] * n
    for i, (label, refs) in enumerate(scaffold):
        if i == 0:
            for c in range(n):
                full_texts[c] += f"{label}\n"
            continue

        n_fields = 1 if i == 1 else (2 if i == 2 else 3)
        partial = [label] * n
        for k in range(n_fields):
            ref = refs[k]
            batch = [prompt + full_texts[c] + partial[c] + f" {ref}"
                     for c in range(n)]
            outputs = llm.generate(batch, params)
            for c in range(n):
                text = outputs[c].outputs[0].text
                m = num_re.match(text)
                if m is None or _parse_first_float(text) is None:
                    vstr = fallback[k]
                else:
                    vstr = m.group(1)
                partial[c] += f" {ref} {vstr}"
        for c in range(n):
            full_texts[c] += partial[c] + "\n"

    return full_texts


def build_template_fh_regex(scaffold):
    """Regex matching exactly one template_fh conformer of `scaffold`.

    Atom labels and reference indices are literal; numeric value positions are
    ``-?\\d+\\.\\d+``. The output ends with a final newline. Used as the
    structured-output constraint for grammar-constrained scaffolded generation.
    """
    import re as _re
    val = r"-?\d+\.\d+"
    parts = []
    for i, (label, refs) in enumerate(scaffold):
        if i == 0:
            parts.append(_re.escape(label))
            continue
        n_fields = 1 if i == 1 else (2 if i == 2 else 3)
        line = "\n" + _re.escape(label)
        for k in range(n_fields):
            line += f" {refs[k]} {val}"
        parts.append(line)
    return "".join(parts) + "\n"


def generate_grammar_zmat(llm, prompts, scaffolds, ns,
                          temperature=0.6, top_p=0.95, max_tokens=4096):
    """Single-pass scaffolded template_fh generation via regex-constrained decoding.

    Builds one regex per molecule from its scaffold (labels + refs literal,
    numeric values free) and submits all (mol, conformer-slot) requests in a
    single ``llm.generate`` call with per-prompt SamplingParams. vLLM's
    continuous batcher schedules them like free-mode generation; the
    structured-output backend (xgrammar by default) masks the model's logits at
    every step so the scaffold tokens are pinned to the right characters and
    only the numeric digits are freely sampled.

    Compute profile matches free mode (single decode pass per conformer with
    ~600 tokens of output), so wall clock should be close to ``generate_free``
    rather than to ``generate_scaffolded_zmat``.

    Args:
        llm: vLLM LLM instance.
        prompts: list of M inference prompts (one per molecule).
        scaffolds: list of M scaffolds, each as returned by
                   TemplateFH.scaffold_for(smiles).
        ns: list of M per-molecule conformer counts.
        temperature, top_p: sampling.
        max_tokens: per-conformer cap. Drug-sized template_fh conformers run
                    well under 2048; 4096 leaves headroom.

    Returns:
        List of M lists, the m-th of which is the ns[m] generated strings
        for molecule m (each string is one full template_fh conformer).
    """
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    M = len(prompts)
    if not (M == len(scaffolds) == len(ns)):
        raise ValueError("prompts, scaffolds, ns must all have same length")

    sp_list = []
    for scaffold, n in zip(scaffolds, ns):
        regex = build_template_fh_regex(scaffold)
        sp_list.append(SamplingParams(
            temperature=temperature, top_p=top_p, max_tokens=max_tokens, n=n,
            structured_outputs=StructuredOutputsParams(regex=regex),
        ))

    outputs = llm.generate(prompts, sp_list)

    all_gens = []
    for out in outputs:
        all_gens.append([o.text for o in out.outputs])
    return all_gens


def _dp_worker(gpu_ids_str, model_kwargs, prompts, gen_kwargs, result_path):
    """Data-parallel worker: loads model on assigned GPU(s) and generates."""
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
    from vllm import LLM, SamplingParams

    llm = LLM(**model_kwargs)

    max_tokens = gen_kwargs.get("max_new_tokens", 1024)
    temperature = gen_kwargs.get("temperature")
    top_p = gen_kwargs.get("top_p")
    top_k = gen_kwargs.get("top_k")
    n = gen_kwargs.get("n", 1)
    do_sample = gen_kwargs.get("do_sample")
    seed = gen_kwargs.get("seed")

    if do_sample is False or temperature is None:
        sampling_params = SamplingParams(max_tokens=max_tokens, temperature=0, n=n)
    else:
        sp_kwargs = {"max_tokens": max_tokens, "temperature": temperature, "n": n}
        if top_p is not None:
            sp_kwargs["top_p"] = top_p
        if top_k is not None:
            sp_kwargs["top_k"] = top_k
        if seed is not None:
            sp_kwargs["seed"] = seed
        sampling_params = SamplingParams(**sp_kwargs)

    outputs = llm.generate(prompts, sampling_params)
    results = [(o.prompt, [t.text for t in o.outputs]) for o in outputs]

    with open(result_path, "wb") as f:
        pickle.dump(results, f)


class InferenceEngine:
    """Batch inference using vllm.

    Args:
        model_path: Path to model (local or HuggingFace ID).
        tensor_parallel_size: Number of GPUs for tensor parallelism.
            None = auto-detect (all GPUs when dp=1, else 1).
        data_parallel_size: Number of model replicas for data parallelism.
            Each replica runs on ``tensor_parallel_size`` GPUs.
            Requires ``data_parallel_size * tensor_parallel_size`` GPUs total.
        trust_remote_code: Whether to trust remote code (e.g., for Qwen).
        dtype: Model dtype. Default "auto".
        max_model_len: Maximum model context length. None = use model default.
        gpu_memory_utilization: Fraction of GPU memory to use (0-1).
    """

    def __init__(self, model_path, tensor_parallel_size=None,
                 data_parallel_size=1, trust_remote_code=False, dtype="auto",
                 max_model_len=None, gpu_memory_utilization=0.90,
                 quantization=None):
        self.model_path = model_path
        self._data_parallel_size = data_parallel_size

        if data_parallel_size > 1:
            if tensor_parallel_size is None:
                tensor_parallel_size = 1
            self._tensor_parallel_size = tensor_parallel_size

            self._model_kwargs = dict(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            if quantization is not None:
                self._model_kwargs["quantization"] = quantization
            # Honor opt-in eager mode on the data-parallel path too (each DP
            # worker LLM gets these kwargs). Previously only the dp==1 branch
            # read this, so dp>1 silently ran torch.compile + CUDA-graph capture.
            if os.environ.get("STRUCTURE_LLM_ENFORCE_EAGER") == "1":
                self._model_kwargs["enforce_eager"] = True

            parent_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
            if parent_visible:
                all_gpus = [g.strip() for g in parent_visible.split(",")]
            else:
                all_gpus = [str(i) for i in range(
                    data_parallel_size * tensor_parallel_size)]

            needed = data_parallel_size * tensor_parallel_size
            if len(all_gpus) < needed:
                raise ValueError(
                    f"Need {needed} GPUs "
                    f"(dp={data_parallel_size} x tp={tensor_parallel_size}) "
                    f"but only {len(all_gpus)} visible")

            self._gpu_assignments = []
            for i in range(data_parallel_size):
                start = i * tensor_parallel_size
                end = start + tensor_parallel_size
                self._gpu_assignments.append(",".join(all_gpus[start:end]))
        else:
            import torch
            from vllm import LLM

            if tensor_parallel_size is None:
                tensor_parallel_size = torch.cuda.device_count() or 1
            self._tensor_parallel_size = tensor_parallel_size

            kwargs = dict(
                model=model_path,
                tensor_parallel_size=tensor_parallel_size,
                trust_remote_code=trust_remote_code,
                dtype=dtype,
                max_model_len=max_model_len,
                gpu_memory_utilization=gpu_memory_utilization,
            )
            if quantization is not None:
                kwargs["quantization"] = quantization
            # Opt-in eager mode: skips torch.compile + CUDA-graph capture, which
            # dominates engine-init time. Useful for short-lived benchmark loads
            # where peak decode throughput is not the bottleneck.
            if os.environ.get("STRUCTURE_LLM_ENFORCE_EAGER") == "1":
                kwargs["enforce_eager"] = True
            self.llm = LLM(**kwargs)

    def generate(self, prompts, max_new_tokens=1024, temperature=None,
                 top_p=None, top_k=None, n=1, do_sample=None, seed=None):
        """Generate text for a batch of prompts.

        Args:
            prompts: List of prompt strings.
            max_new_tokens: Maximum tokens to generate per prompt.
            temperature: Sampling temperature. None = greedy.
            top_p: Nucleus sampling threshold.
            top_k: Top-k sampling. None = disabled.
            n: Number of sequences to return per prompt.
            do_sample: If False, use greedy decoding (overrides temperature).
            seed: Per-request sampling seed. None (default) = unseeded, which is
                what every published run used -- passing None reproduces the
                historical behaviour exactly. Note that a seed makes sampling
                repeatable only for a fixed vLLM version, hardware and batch
                composition; continuous batching means it is not a guarantee of
                bitwise reproducibility across environments.

        Returns:
            List of (prompt, [generated_texts]) tuples, one per input prompt.
        """
        if self._data_parallel_size > 1:
            return self._generate_dp(
                prompts, max_new_tokens=max_new_tokens,
                temperature=temperature, top_p=top_p, top_k=top_k,
                n=n, do_sample=do_sample, seed=seed)

        from vllm import SamplingParams

        if do_sample is False or temperature is None:
            sampling_params = SamplingParams(
                max_tokens=max_new_tokens, temperature=0, n=n)
        else:
            kwargs = {"max_tokens": max_new_tokens,
                      "temperature": temperature, "n": n}
            if top_p is not None:
                kwargs["top_p"] = top_p
            if top_k is not None:
                kwargs["top_k"] = top_k
            if seed is not None:
                kwargs["seed"] = seed
            sampling_params = SamplingParams(**kwargs)

        outputs = self.llm.generate(prompts, sampling_params)

        results = []
        for output in outputs:
            prompt = output.prompt
            texts = [o.text for o in output.outputs]
            results.append((prompt, texts))

        return results

    def generate_feedback(self, prompt, n=1, temperature=0.6, top_p=0.95,
                          max_atoms=300, max_tokens_per_line=128):
        """Generate feedback_fh conformers. See generate_feedback_fh()."""
        if self._data_parallel_size > 1:
            raise NotImplementedError(
                "feedback_fh generation does not support data parallelism")
        return generate_feedback_fh(
            self.llm, prompt, n=n, temperature=temperature, top_p=top_p,
            max_atoms=max_atoms, max_tokens_per_line=max_tokens_per_line)

    def generate_scaffolded(self, prompt, labels, n=1, temperature=0.6,
                            top_p=0.95, max_tokens_per_atom=32):
        """Generate a labeled_xyz conformer. See generate_scaffolded_xyz()."""
        if self._data_parallel_size > 1:
            raise NotImplementedError(
                "scaffolded labeled_xyz generation does not support data "
                "parallelism")
        return generate_scaffolded_xyz(
            self.llm, prompt, labels, n=n, temperature=temperature,
            top_p=top_p, max_tokens_per_atom=max_tokens_per_atom)

    def generate_scaffolded_zmat(self, prompt, scaffold, n=1, temperature=0.6,
                                 top_p=0.95, max_tokens_per_value=16):
        """Generate a template_fh conformer. See generate_scaffolded_zmat()."""
        if self._data_parallel_size > 1:
            raise NotImplementedError(
                "scaffolded template_fh generation does not support data "
                "parallelism")
        return generate_scaffolded_zmat(
            self.llm, prompt, scaffold, n=n, temperature=temperature,
            top_p=top_p, max_tokens_per_value=max_tokens_per_value)

    def generate_grammar_zmat(self, prompts, scaffolds, ns,
                              temperature=0.6, top_p=0.95, max_tokens=4096):
        """Grammar-constrained scaffolded template_fh generation.
        See generate_grammar_zmat()."""
        if self._data_parallel_size > 1:
            raise NotImplementedError(
                "grammar-constrained template_fh generation does not yet "
                "support data parallelism")
        return generate_grammar_zmat(
            self.llm, prompts, scaffolds, ns,
            temperature=temperature, top_p=top_p, max_tokens=max_tokens)

    def _generate_dp(self, prompts, **gen_kwargs):
        """Generate with data parallelism across multiple model replicas."""
        import shutil

        ctx = mp.get_context("spawn")
        dp = self._data_parallel_size
        tmp_dir = tempfile.mkdtemp(prefix="vllm_dp_")

        chunk_size = (len(prompts) + dp - 1) // dp
        chunks = []
        for i in range(dp):
            s = i * chunk_size
            e = min(s + chunk_size, len(prompts))
            if s < len(prompts):
                chunks.append(prompts[s:e])

        print(f"Data-parallel inference: {len(chunks)} workers, "
              f"{len(prompts)} prompts "
              f"({' / '.join(str(len(c)) for c in chunks)} split)")

        result_paths = [os.path.join(tmp_dir, f"worker_{i}.pkl")
                        for i in range(len(chunks))]

        processes = []
        for i, chunk in enumerate(chunks):
            p = ctx.Process(
                target=_dp_worker,
                args=(self._gpu_assignments[i], self._model_kwargs,
                      chunk, gen_kwargs, result_paths[i]),
            )
            p.start()
            processes.append(p)

        for p in processes:
            p.join()

        for i, p in enumerate(processes):
            if p.exitcode != 0:
                shutil.rmtree(tmp_dir, ignore_errors=True)
                raise RuntimeError(
                    f"Data-parallel worker {i} "
                    f"(GPUs {self._gpu_assignments[i]}) "
                    f"exited with code {p.exitcode}")

        all_results = []
        for path in result_paths:
            with open(path, "rb") as f:
                all_results.extend(pickle.load(f))

        shutil.rmtree(tmp_dir, ignore_errors=True)
        return all_results
