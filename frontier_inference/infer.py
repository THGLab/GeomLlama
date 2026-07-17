import asyncio
from asyncio_throttle import Throttler
import litellm

from prompts import XYZ_SYSTEM_PROMPT, ZMAT_SYSTEM_PROMPT, make_user_prompt


# Silence litellm's verbose logging
litellm.suppress_debug_info = True


async def generate_geometry(
    smiles: str,
    model: str = "gpt-4o-mini",
    fmt: str = "xyz",
    temperature: float = 0.0,
    seed: int = 42,
    reasoning_effort: str | None = None,
    thinking_budget: int | None = None,
    throttler: Throttler | None = None,
) -> dict:
    """Call an LLM to generate a molecular geometry from a SMILES string.

    Works with OpenAI, Anthropic (Claude), and Google (Gemini) models via
    litellm. Pass model names like "gpt-4o-mini", "o3",
    "claude-opus-4-5-20251001", or "gemini/gemini-2.5-pro".

    If `reasoning_effort` is one of "low", "medium", "high", extended thinking
    is enabled on providers that support it (OpenAI o-series, Claude 4.x,
    Gemini 2.5).

    Returns a dict with keys: smiles, model, format, raw_response, reasoning.
    """
    system_prompt = XYZ_SYSTEM_PROMPT if fmt == "xyz" else ZMAT_SYSTEM_PROMPT
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": make_user_prompt(smiles)},
    ]

    kwargs = dict(
        model=model,
        messages=messages,
        temperature=temperature,
        seed=seed,
        drop_params=True,
    )
    if reasoning_effort is not None:
        kwargs["reasoning_effort"] = reasoning_effort
    if thinking_budget is not None:
        # Direct token-level control. Overrides reasoning_effort on providers
        # that support both (Anthropic, Gemini). 0 disables thinking entirely.
        kwargs["thinking"] = {
            "type": "enabled" if thinking_budget > 0 else "disabled",
            "budget_tokens": thinking_budget,
        }

    async def _call(call_kwargs):
        if throttler:
            async with throttler:
                return await litellm.acompletion(**call_kwargs)
        return await litellm.acompletion(**call_kwargs)

    try:
        response = await _call(kwargs)
    except litellm.BadRequestError as e:
        # Some reasoning-only models (e.g. GPT-5.x, o-series) reject temperature/seed
        # outright. litellm's drop_params only helps for models in its registry, so
        # for brand-new models we fall back by stripping whichever param the API
        # named in the error.
        msg = str(e).lower()
        fallback = {k: v for k, v in kwargs.items()
                    if not ((k == "temperature" and "temperature" in msg) or
                            (k == "seed" and "seed" in msg))}
        if fallback == kwargs:
            raise
        response = await _call(fallback)

    message = response.choices[0].message
    raw = (message.content or "").strip()
    reasoning = getattr(message, "reasoning_content", None)

    usage = getattr(response, "usage", None)
    usage_dict = {
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
        "total_tokens": getattr(usage, "total_tokens", None),
    } if usage else None

    try:
        cost = litellm.completion_cost(completion_response=response)
    except Exception:
        cost = None

    return {
        "smiles": smiles,
        "model": model,
        "format": fmt,
        "raw_response": raw,
        "reasoning": reasoning,
        "usage": usage_dict,
        "cost": cost,
    }


async def generate_batch(
    smiles_list: list[str],
    model: str = "gpt-4o-mini",
    fmt: str = "xyz",
    temperature: float = 0.0,
    seed: int = 42,
    reasoning_effort: str | None = None,
    thinking_budget: int | None = None,
    max_concurrency: int = 10,
    requests_per_second: float = 5,
    requests_per_minute: float | None = None,
) -> list[dict]:
    """Generate geometries for a batch of SMILES in parallel.

    Use `requests_per_minute` for tight rate limits like the free Gemini tier (~10 RPM).
    """
    if requests_per_minute is not None:
        throttler = Throttler(rate_limit=requests_per_minute, period=60.0)
    else:
        throttler = Throttler(rate_limit=requests_per_second, period=1.0)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _limited(smi):
        async with semaphore:
            return await generate_geometry(
                smi, model=model, fmt=fmt,
                temperature=temperature, seed=seed,
                reasoning_effort=reasoning_effort,
                thinking_budget=thinking_budget,
                throttler=throttler,
            )

    tasks = [_limited(smi) for smi in smiles_list]
    return await asyncio.gather(*tasks, return_exceptions=True)


async def generate_batch_stream(
    smiles_list: list[str],
    model: str = "gpt-4o-mini",
    fmt: str = "xyz",
    temperature: float = 0.0,
    seed: int = 42,
    reasoning_effort: str | None = None,
    thinking_budget: int | None = None,
    max_concurrency: int = 10,
    requests_per_second: float = 5,
    requests_per_minute: float | None = None,
):
    """Like generate_batch, but yields (index, result_or_exception) as each
    SMILES finishes, rather than returning all results at the end.

    `index` is the position in `smiles_list`, so the caller can pair the result
    back to its entry (order of completion is NOT the input order). Exceptions
    are yielded rather than raised, mirroring generate_batch's return_exceptions.
    """
    if requests_per_minute is not None:
        throttler = Throttler(rate_limit=requests_per_minute, period=60.0)
    else:
        throttler = Throttler(rate_limit=requests_per_second, period=1.0)
    semaphore = asyncio.Semaphore(max_concurrency)

    async def _limited(i, smi):
        async with semaphore:
            try:
                return i, await generate_geometry(
                    smi, model=model, fmt=fmt,
                    temperature=temperature, seed=seed,
                    reasoning_effort=reasoning_effort,
                    thinking_budget=thinking_budget,
                    throttler=throttler,
                )
            except Exception as e:  # noqa: BLE001 - surfaced to caller as a value
                return i, e

    tasks = [asyncio.create_task(_limited(i, smi))
             for i, smi in enumerate(smiles_list)]
    for coro in asyncio.as_completed(tasks):
        yield await coro
