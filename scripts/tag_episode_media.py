"""Migrate an existing GIF and its receipt without rerunning or altering inference."""

import argparse
import json
from pathlib import Path

from flyper_wolf.media_identity import tag_gif
from flyper_wolf.seeding import normalize_master_seed


def tag_receipt(path, master_seed):
    path = Path(path)
    report = json.loads(path.read_text(encoding="utf-8"))
    master_seed = normalize_master_seed(master_seed)
    if "master_seed" in report and report["master_seed"] != master_seed:
        raise ValueError("Receipt already identifies a different episode")
    gif = path.with_name(path.name.removesuffix(".provenance.json") + ".gif") if path.name.endswith(".provenance.json") else path.with_suffix(".gif")
    before_size = gif.stat().st_size
    previous_hash = report.get("gif_sha256", report.get("outputs", {}).get(gif.name, {}).get("sha256"))
    if previous_hash:
        from flyper_wolf.neural_recording import _hash
        if _hash(gif) != previous_hash:
            raise ValueError(f"Existing GIF differs from its receipt: {gif}")
    annotation = tag_gif(gif, master_seed)
    report.update({key: annotation[key] for key in ("master_seed", "seed_scheme")})
    # Keep historical source hashes: annotation does not rewrite the source run.
    report["seed_identity_annotation"] = annotation
    if "gif_sha256" in report:
        report["gif_sha256"] = annotation["gif_sha256"]
    if "gif_size_bytes" in report:
        report["gif_size_bytes"] = gif.stat().st_size
    if gif.name in report.get("outputs", {}):
        report["outputs"][gif.name].update(sha256=annotation["gif_sha256"], bytes=gif.stat().st_size)
    path.write_text(json.dumps(report, indent=2, allow_nan=False)+"\n", encoding="utf-8")
    return {"receipt": str(path), "gif": str(gif), "size_delta": gif.stat().st_size-before_size, **annotation}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("receipt", type=Path, nargs="+")
    parser.add_argument("--master-seed", required=True)
    args = parser.parse_args()
    for receipt in args.receipt:
        print(json.dumps(tag_receipt(receipt, args.master_seed)))
