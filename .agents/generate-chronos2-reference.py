#!/usr/bin/env python3
"""Generate compact Chronos-2 forecast fixtures from pinned official code.

Required environment variables:
  CHRONOS_SOURCE       checkout of amazon-science/chronos-forecasting at SOURCE_COMMIT
  CHRONOS_CHECKPOINT   directory containing config.json and model.safetensors

Usage:
  python .agents/generate-chronos2-reference.py tests/testthat/fixtures/chronos2
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SOURCE_COMMIT = "8589d1988e9676817548e9626738ff06b6ca6370"
CHECKPOINT_ID = "amazon/chronos-2"
CHECKPOINT_REVISION = "29ec3766d36d6f73f0696f85560a422f50e8498c"
QUANTILE_LEVELS = [0.01] + [round(0.05 * k, 2) for k in range(1, 20)] + [0.99]
OUTPUT_PATCH_SIZE = 16


def assert_source_pin(source: Path) -> None:
  actual = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=source, text=True
  ).strip()
  if actual != SOURCE_COMMIT:
    raise RuntimeError(f"Chronos source is {actual}; expected {SOURCE_COMMIT}")


def series(kind: str, length: int, rng: np.random.Generator) -> np.ndarray:
  x = np.arange(1, length + 1, dtype=np.float32)
  if kind == "trend_seasonal":
    return (100.0 + 0.5 * x + 10.0 * np.sin(x * 2.0 * np.pi / 24.0)).astype(np.float32)
  if kind == "mixed_sign":
    # Crosses zero and ends below it, so the negative branch of the sinh
    # inverse has a reference attached; every other case here is positive.
    return (3.0 * np.sin(x * 2.0 * np.pi / 16.0) - 0.05 * x).astype(np.float32)
  if kind == "flat":
    # Zero variance reaches the instance-norm epsilon clamp, which a smooth
    # series never does.
    return np.full(length, 7.5, dtype=np.float32)
  if kind == "promo_driver":
    return (x % 12 < 3).astype(np.float32)
  if kind == "promo_target":
    driver = (x % 12 < 3).astype(np.float32)
    return (100.0 + 0.4 * x + 25.0 * driver + rng.normal(0, 1.5, length)).astype(np.float32)
  raise ValueError(f"Unknown series generator: {kind}")


def fixture_cases() -> list[dict]:
  # Chronos-2 accepts any context length -- it pads internally -- so unlike the
  # Toto fixtures these need not be patch-aligned.
  return [
    {"name": "typical", "rows": [{"kind": "trend_seasonal", "length": 256}],
     "horizon": 24},
    {"name": "short_context", "rows": [{"kind": "trend_seasonal", "length": 48}],
     "horizon": 8},
    {"name": "long_horizon", "rows": [{"kind": "trend_seasonal", "length": 512}],
     "horizon": 96},
    {"name": "mixed_sign", "rows": [{"kind": "mixed_sign", "length": 128}],
     "horizon": 16},
    {"name": "flat_context", "rows": [{"kind": "flat", "length": 96}],
     "horizon": 8},
    # Two independent series in one call: `group_ids` keeps them apart, so each
    # must match what it would produce alone.
    {"name": "independent_batch",
     "rows": [{"kind": "trend_seasonal", "length": 192},
              {"kind": "mixed_sign", "length": 192}],
     "horizon": 12, "groups": [0, 1]},
    # One task: a target plus a covariate whose future is known. This is the
    # case no contract-1.0 architecture can express.
    {"name": "future_covariate",
     "rows": [{"kind": "promo_target", "length": 192},
              {"kind": "promo_driver", "length": 192}],
     "horizon": 16, "groups": [0, 0], "targets": [True, False],
     "future": [None, "promo_driver_future"]},
  ]


def main() -> None:
  if len(sys.argv) != 2:
    raise SystemExit("usage: generate-chronos2-reference.py OUTPUT_DIR")
  source = Path(os.environ["CHRONOS_SOURCE"]).resolve()
  checkpoint = Path(os.environ["CHRONOS_CHECKPOINT"]).resolve()
  output_dir = Path(sys.argv[1]).resolve()
  assert_source_pin(source)
  sys.path.insert(0, str(source / "src"))

  import torch  # pylint: disable=import-outside-toplevel
  from chronos.chronos2.model import Chronos2Model  # pylint: disable=import-error,import-outside-toplevel

  model = Chronos2Model.from_pretrained(str(checkpoint)).eval()
  output_dir.mkdir(parents=True, exist_ok=True)

  for case in fixture_cases():
    rng = np.random.default_rng(20260905)
    rows = [series(spec["kind"], spec["length"], rng) for spec in case["rows"]]
    width = max(len(row) for row in rows)
    padded = np.full((len(rows), width), np.nan, dtype=np.float32)
    for index, row in enumerate(rows):
      padded[index, width - len(row):] = row

    horizon = case["horizon"]
    num_output_patches = int(np.ceil(horizon / OUTPUT_PATCH_SIZE))
    targets = case.get("targets", [True] * len(rows))
    group_ids = case.get("groups", list(range(len(rows))))

    future = None
    future_values: list[list[float]] = []
    if "future" in case:
      future = np.full((len(rows), horizon), np.nan, dtype=np.float32)
      for index, spec in enumerate(case["future"]):
        if spec is None:
          future_values.append([])
          continue
        ahead = (np.arange(len(rows[index]) + 1, len(rows[index]) + horizon + 1) % 12 < 3)
        future[index] = ahead.astype(np.float32)
        future_values.append(future[index].tolist())
    else:
      future_values = [[] for _ in rows]

    with torch.no_grad():
      out = model(
        context=torch.tensor(padded),
        group_ids=torch.tensor(group_ids, dtype=torch.long),
        future_covariates=None if future is None else torch.tensor(future),
        num_output_patches=num_output_patches,
      )
    quantiles = out.quantile_preds[:, :, :horizon].numpy()   # (rows, q, h)

    context_files = []
    for index, row in enumerate(rows, start=1):
      context_file = f"{case['name']}-context-{index}.f32"
      np.asarray(row, dtype="<f4").tofile(output_dir / context_file)
      context_files.append(context_file)

    payload = {
      "schema_version": 1,
      "name": case["name"],
      "source_commit": SOURCE_COMMIT,
      "model_id": CHECKPOINT_ID,
      "revision": CHECKPOINT_REVISION,
      "reference": {"numpy": np.__version__, "torch": torch.__version__},
      "context_specs": case["rows"],
      "context_files": context_files,
      "horizon": horizon,
      "group_ids": [str(g) for g in group_ids],
      "targets": targets,
      "future": future_values,
      "quantile_levels": QUANTILE_LEVELS,
      # One entry per *target* row, matching what the engine returns.
      "expected_quantiles": [
        quantiles[index].T.tolist()
        for index, is_target in enumerate(targets) if is_target
      ],
      "atol": 1e-4,
      "rtol": 1e-5,
    }
    (output_dir / f"{case['name']}.json").write_text(json.dumps(payload))
    print(f"wrote {case['name']}: {sum(targets)} target(s) x {horizon}")


if __name__ == "__main__":
  main()
