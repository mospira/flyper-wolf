"""Check a generated episode against the documented compatibility reference."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import yaml


def check(run, reference):
    run = Path(run)
    expected = json.loads(Path(reference).read_text(encoding="utf-8"))
    metadata = json.loads((run / "metadata.json").read_text(encoding="utf-8"))
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    config = yaml.safe_load((run / "resolved-config.yaml").read_text(encoding="utf-8"))
    if metadata["status"] != "complete":
        raise ValueError("Run is not complete")
    for name, digest in metadata["output_sha256"].items():
        path = run / name
        if not path.resolve().is_relative_to(run.resolve()):
            raise ValueError("Invalid artifact path in run metadata")
        with path.open("rb") as handle:
            if hashlib.file_digest(handle, "sha256").hexdigest() != digest:
                raise ValueError(f"Output checksum mismatch: {name}")
    actual = metadata["runtime"]
    if not actual["frozen_parameters_unchanged"]:
        raise ValueError("Frozen parameters changed")
    for group, values in expected["identity"].items():
        for key, value in values.items():
            if actual[group].get(key) != value:
                raise ValueError(f"Reference identity differs: {group}.{key}")
    for group, values in expected.get("configuration", {}).items():
        for key, value in values.items():
            if config[group].get(key) != value:
                raise ValueError(f"Reference configuration differs: {group}.{key}")
    for key, item in expected["metrics"].items():
        if not math.isclose(metrics[key], item["value"], rel_tol=0, abs_tol=item["absolute_tolerance"]):
            raise ValueError(f"Reference metric differs: {key}: {metrics[key]} vs {item['value']}")
    with (run / "trace.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != metrics["circuit_steps"]:
        raise ValueError("Trace length differs from circuit-step count")
    for tick, row in enumerate(rows):
        if int(row["tick"]) != tick or row["prediction_updated"] != str(tick >= 20):
            raise ValueError(f"Trace cadence differs at tick {tick}")
        if tick < 20 and (float(row["action_yaw"]) != 0 or float(row["action_pitch"]) != 0):
            raise ValueError("Startup action is nonzero")
    return {"status": "passed", "reference": expected["name"], "ticks": len(rows),
            "time_on_target_pct": metrics["time_on_target_pct"],
            "scope": "Compatibility check, not an independent research benchmark"}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--reference", type=Path, default=Path(__file__).resolve().parents[1] / "configs/reference-fw-est342-fw-cont24.json")
    args = parser.parse_args()
    try:
        print(json.dumps(check(args.run, args.reference), indent=2))
    except (ValueError, KeyError, OSError) as error:
        parser.exit(1, f"Reference check failed: {error}\n")
