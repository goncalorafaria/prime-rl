"""Materialize node addresses in the search smoke config, then launch PrimeRL."""

import argparse
import os
import tomllib
from pathlib import Path

import tomli_w

from prime_rl.shardcast.protocol import validate_run_id


def training_config(environment):
    run_id = validate_run_id(environment["RUN_ID"])
    gateway = environment["GATEWAY_URL"].rstrip("/")
    root = Path(__file__).resolve().parents[3]
    config = tomllib.loads((root / "examples/shardcast-search/search-2b.toml").read_text())
    config["output_dir"] = environment.get("OUTPUT_DIR", f"outputs/{run_id}")
    if "BASE_MODEL" in environment:
        config["model"]["name"] = environment["BASE_MODEL"]
    client = config["orchestrator"]["model"]["client"]
    client["base_url"] = [gateway + "/v1"]
    client["shardcast"].update(
        registry=environment["REGISTRY"], run_id=run_id, origin_host=environment["ADVERTISE_HOST"]
    )
    tools = config["orchestrator"]["train"]["source"][0]["legacy"]["args"]
    for service in ("search", "terminal", "judge"):
        tools[f"{service}_server_url"] = f"{gateway}/{service}"
    return config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write-config", type=Path, help="Write the resolved launch input without starting training")
    args, overrides = parser.parse_known_args()
    config = training_config(os.environ)
    destination = args.write_config or Path(config["output_dir"]) / "launch.toml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(tomli_w.dumps(config))
    if args.write_config:
        print(destination)
        return
    os.execvp("uv", ["uv", "run", "--no-sync", "rl", "@", str(destination), *overrides])


if __name__ == "__main__":
    main()
