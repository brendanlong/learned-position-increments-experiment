"""Save the wandb histories the report plots (val loss, largest delta)."""

import json
import sys

import wandb

RUNS = {
    "rope": "ycqqcknn",
    "shared": "4zee13hf",
    "perlayer_s42": "zp4muno6",
    "perlayer_s1": "qi0epy7e",
    "perlayer_s2": "3866xiyd",
    "control": "rrzpmhro",
    "blog_perlayer": "0eqetsep",
    "big_rope": "ac95615f",
    "big_s42": "0669d9cc",
    "big_s1": "3adb9f11",
}


def main() -> None:
    api = wandb.Api()
    out = {}
    for name, rid in RUNS.items():
        run = api.run(f"brendanlong-com/byte-pe/{rid}")
        rows = run.scan_history(keys=["_step", "val/loss"])
        val = [(r["_step"], r["val/loss"]) for r in rows]
        rows = run.scan_history(keys=["_step", "delta/max"])
        dmax = [(r["_step"], r["delta/max"]) for r in rows]
        out[name] = {"id": rid, "name": run.name, "val_loss": val, "delta_max": dmax}
        print(name, len(val), len(dmax))
    with open(sys.argv[1], "w") as f:
        json.dump(out, f)


if __name__ == "__main__":
    main()
