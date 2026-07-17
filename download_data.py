#!/usr/bin/env python3
"""Download GeomLlama data archives from figshare.

Usage:
    python download_data.py --datasets       # Test/eval JSON + pickles
    python download_data.py --sft            # JSONL training data
    python download_data.py --results        # Frontier inference + SmileyLlama results
    python download_data.py --alpaca         # Alpaca instruction mix (from upstream)
    python download_data.py --all            # Everything
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request
import zipfile

URLS = {
    "datasets": "FIGSHARE_URL_DATASETS",
    "sft": "FIGSHARE_URL_SFT",
    "results": "FIGSHARE_URL_RESULTS",
}

# Every archive stores repo-root-relative paths ("data/qm9/test_set.json",
# "cache/1k_smiles_ro4/xtb_rdkit.pkl"), so they all extract at the repo root and
# reproduce the layout verbatim. The results archive spans BOTH results/ and a
# top-level cache/, which synth reads (config.CACHE_DIR = {REPO}/cache/<dataset>).
EXTRACT_TO = {
    "datasets": ".",
    "sft": ".",
    "results": ".",
}

# The hybrid configs (configs/examples/geom_8b_hybrid_*.yml) mix in the Alpaca
# instruction set to preserve language ability. It is not redistributed in the
# figshare archives: Alpaca is CC BY-NC 4.0, and bundling it would make those
# archives unable to carry an open license. It is fetched from upstream instead.
#
# Pinned to the commit that released the file (2023-03-13); the content is
# verified by hash, so a moved tag or an upstream edit is caught rather than
# silently changing the training data.
ALPACA_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/"
    "f13496211289def0ff88ae673389ae14a818b4b3/alpaca_data.json"
)
ALPACA_SHA256 = "2eddafc6b977608d778aaab8dfc7e50e547b3af9826dfb9e909d9fc362e4a419"
ALPACA_RECORDS = 52002
ALPACA_DEST = os.path.join("data", "alpaca", "alpaca.jsonl")


def download_and_extract(name, url, extract_to):
    if url.startswith("FIGSHARE_URL"):
        print(f"'{name}' is not available yet.\n"
              f"\n"
              f"The figshare archives are being uploaded and will be available\n"
              f"shortly; this script will be updated with their DOIs. In the\n"
              f"meantime the code, the training configs, and `--alpaca` all work.\n"
              f"If you need this data now, please open an issue at\n"
              f"https://github.com/THGLab/GeomLlama/issues -- we would rather send\n"
              f"it to you directly than have you wait.")
        return False

    zip_path = f"{name}.zip"
    print(f"Downloading {name}...")
    urllib.request.urlretrieve(url, zip_path)

    print(f"Extracting to {extract_to}/...")
    os.makedirs(extract_to, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_to)

    os.remove(zip_path)
    print(f"Done: {name}")
    return True


def download_alpaca():
    """Fetch the Alpaca instruction set and write it as JSONL.

    Upstream ships a JSON array; the trainer expects one record per line.
    """
    if os.path.exists(ALPACA_DEST):
        print(f"Already present: {ALPACA_DEST}")
        return True

    print("Downloading alpaca (from tatsu-lab/stanford_alpaca)...")
    with urllib.request.urlopen(ALPACA_URL) as resp:
        raw = resp.read()

    digest = hashlib.sha256(raw).hexdigest()
    if digest != ALPACA_SHA256:
        print(f"ERROR: alpaca checksum mismatch.\n"
              f"  expected: {ALPACA_SHA256}\n"
              f"  got:      {digest}\n"
              f"Refusing to write: upstream content changed, and training on it "
              f"would not reproduce the paper.")
        return False

    records = json.loads(raw)
    if len(records) != ALPACA_RECORDS:
        print(f"ERROR: expected {ALPACA_RECORDS} alpaca records, got {len(records)}")
        return False

    os.makedirs(os.path.dirname(ALPACA_DEST), exist_ok=True)
    with open(ALPACA_DEST, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Done: alpaca ({len(records)} records -> {ALPACA_DEST})")
    print("Note: Alpaca is CC BY-NC 4.0 (tatsu-lab/stanford_alpaca), derived "
          "from OpenAI text-davinci-003 output. Research use only.")
    return True


def main():
    parser = argparse.ArgumentParser(
        description="Download GeomLlama data archives from figshare")
    parser.add_argument("--datasets", action="store_true",
                        help="Download test/eval datasets (JSON + pickles)")
    parser.add_argument("--sft", action="store_true",
                        help="Download SFT training data (JSONL)")
    parser.add_argument("--results", action="store_true",
                        help="Download saved results (frontier inference + SmileyLlama)")
    parser.add_argument("--alpaca", action="store_true",
                        help="Download the Alpaca instruction mix from upstream "
                             "(needed by the GEOM hybrid configs)")
    parser.add_argument("--all", action="store_true",
                        help="Download everything")
    args = parser.parse_args()

    if not any([args.datasets, args.sft, args.results, args.alpaca, args.all]):
        parser.print_help()
        sys.exit(1)

    targets = []
    if args.all or args.datasets:
        targets.append("datasets")
    if args.all or args.sft:
        targets.append("sft")
    if args.all or args.results:
        targets.append("results")

    ok = True
    for name in targets:
        ok &= download_and_extract(name, URLS[name], EXTRACT_TO[name])

    if args.all or args.alpaca:
        ok &= download_alpaca()

    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
