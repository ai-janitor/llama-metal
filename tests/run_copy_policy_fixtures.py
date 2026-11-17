#!/usr/bin/env python3
"""Exercise frozen P13.18 TSVs through the public receipt-validator CLI."""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "metal-phase"
VALIDATOR = ROOT / "tools" / "metal-jobs" / "host-phases" / "verify_phase_receipts.py"
IDENTITY = "('decode', 1, 1)"
TIMING = f"sample={IDENTITY} gpu=6 host_nonoverlap=2 uncovered=0 overlap=4\n"


def case(name: str, args: list[str], code: int, text: str) -> tuple[str, list[str], int, str]:
    return name, args, code, text


CASES = (
    case("cpu-valid", ["copy-policy-cpu-valid.tsv", "valid", "--copy-policy", "cpu"], 0,
         f"sample={IDENTITY} copy_policy=cpu raw_logits_copy=1 sample_token_copy=0\n{TIMING}"),
    case("backend-valid", ["copy-policy-backend-valid.tsv", "valid", "--copy-policy", "backend-sampled"], 0,
         f"sample={IDENTITY} copy_policy=backend-sampled raw_logits_copy=0 sample_token_copy=1\n{TIMING}"),
    case("backend-extra-raw", ["copy-policy-backend-extra-raw.tsv", "valid", "--copy-policy", "backend-sampled"], 1,
         "copy policy backend-sampled requires raw_logits_copy=0 and sample_token_copy=1"),
    case("backend-missing-sample", ["copy-policy-backend-missing-sample.tsv", "valid", "--copy-policy", "backend-sampled"], 1,
         "copy policy backend-sampled requires raw_logits_copy=0 and sample_token_copy=1"),
    case("backend-duplicate-sample", ["copy-policy-backend-duplicate-sample.tsv", "valid", "--copy-policy", "backend-sampled"], 1,
         "copy policy backend-sampled requires raw_logits_copy=0 and sample_token_copy=1"),
    case("backend-zero-payload", ["copy-policy-backend-zero-sample-payload.tsv", "valid", "--copy-policy", "backend-sampled"], 1,
         "requires positive sample_token_copy payload"),
    case("cpu-unexpected-sample", ["copy-policy-cpu-unexpected-sample.tsv", "valid", "--copy-policy", "cpu"], 1,
         "copy policy cpu requires raw_logits_copy=1 and sample_token_copy=0"),
    case("legacy-no-policy", ["copy-policy-backend-missing-sample.tsv", "valid"], 0, TIMING),
    case("unknown-policy", ["copy-policy-cpu-valid.tsv", "valid", "--copy-policy", "unknown"], 2,
         "invalid choice: 'unknown'"),
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite existing output directory: {args.out}")
    args.out.mkdir(parents=True)
    failures = []
    for name, command_args, wanted_code, wanted_text in CASES:
        command = [sys.executable, "-I", str(VALIDATOR), str(FIXTURES / command_args[0]), *command_args[1:]]
        run = subprocess.run(command, text=True, capture_output=True, check=False)
        receipt = f"command={' '.join(command)}\nexit={run.returncode}\nstdout:\n{run.stdout}stderr:\n{run.stderr}"
        (args.out / f"{name}.txt").write_text(receipt, encoding="utf-8")
        observed = run.stdout if wanted_code == 0 else run.stderr
        ok = run.returncode == wanted_code and (run.stdout == wanted_text if wanted_code == 0 else wanted_text in observed)
        print(f"case={name} exit={run.returncode} assertion={'PASS' if ok else 'FAIL'}")
        if not ok:
            failures.append(name)
    if failures:
        print(f"check-fixtures FAIL cases={','.join(failures)}", file=sys.stderr)
        return 1
    print(f"check-fixtures PASS cases={len(CASES)} out={args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
