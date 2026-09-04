#!/usr/bin/env python3
"""Generate compact Toto 2.0 4M forecast fixtures from pinned official code.

Required environment variables:
  TOTO_SOURCE       checkout of DataDog/toto at SOURCE_COMMIT
  TOTO_CHECKPOINT   directory containing config.json and model.safetensors

Usage:
  python .agents/generate-toto-reference.py tests/testthat/fixtures/toto
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

SOURCE_COMMIT = "46bd92adeeef6b3c2afb21008659f607fa203e38"
CHECKPOINT_ID = "Datadog/Toto-2.0-4m"
CHECKPOINT_REVISION = "8306a9801cf98c0f5ffe4b2dcc8f496e616d84d9"
QUANTILE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
PATCH_SIZE = 32

# This port pins the raw Toto2Model.forecast() knobs. The GluonTS wrapper
# defaults differ (block decoding, a scaler fallback, and a real-space quantile
# cap), and fixtures generated under those settings are not comparable.
FORECAST_KWARGS = {
  "decode_block_size": 0,
  "scaler_fallback_min_obs": 0,
  "quantile_real_cap_k": 0.0,
  "has_missing_values": True,
}


def assert_source_pin(source: Path) -> None:
  actual = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], cwd=source, text=True
  ).strip()
  if actual != SOURCE_COMMIT:
    raise RuntimeError(f"Toto source is {actual}; expected {SOURCE_COMMIT}")


def generated_context(spec: dict) -> np.ndarray:
  kind = spec["kind"]
  length = int(spec["length"])
  x = np.arange(1, length + 1, dtype=np.float32)
  if kind == "trend_seasonal":
    return (100.0 + 0.5 * x + 10.0 * np.sin(x * 2.0 * np.pi / 24.0)).astype(np.float32)
  if kind == "long_trend_seasonal":
    return (50.0 + 0.01 * x + 2.0 * np.sin(x * 2.0 * np.pi / 48.0)).astype(np.float32)
  if kind == "mixed_sign":
    # Crosses zero repeatedly and ends below it. Every other fixture here is
    # strictly positive, so without this one the negative branch of the sinh
    # inverse and the quantile sort would have no reference attached.
    return (3.0 * np.sin(x * 2.0 * np.pi / 16.0) - 0.05 * x).astype(np.float32)
  if kind == "flat":
    # Zero variance: the causal scaler clamps to `minimum_scale` here, which is
    # the branch a smooth synthetic series never reaches.
    return np.full(length, 7.5, dtype=np.float32)
  raise ValueError(f"Unknown context generator: {kind}")


def fixture_cases() -> list[dict]:
  # Upstream requires patch-aligned contexts: `forecast()` reduces the mask in
  # chunks of patch_size and raises otherwise. The R port left-pads instead, so
  # every length here is a multiple of 32 and the padding path is covered by
  # the batch_agreement case, whose two series differ in length.
  return [
    {"name": "typical", "contexts": [{"kind": "trend_seasonal", "length": 256}],
     "horizon": 24},
    {"name": "short_context", "contexts": [{"kind": "trend_seasonal", "length": 64}],
     "horizon": 8},
    {"name": "long_horizon", "contexts": [{"kind": "long_trend_seasonal", "length": 512}],
     "horizon": 96},
    {"name": "mixed_sign", "contexts": [{"kind": "mixed_sign", "length": 128}],
     "horizon": 16},
    {"name": "flat_context", "contexts": [{"kind": "flat", "length": 96}],
     "horizon": 8},
    {"name": "batch_agreement",
     "contexts": [{"kind": "trend_seasonal", "length": 192},
                  {"kind": "long_trend_seasonal", "length": 96}],
     "horizon": 12},
  ]


def materialize_context(spec: dict) -> np.ndarray:
  if "values" in spec:
    return np.asarray(spec["values"], dtype=np.float32)
  return generated_context(spec)


def forecast(model, torch, contexts: list[np.ndarray], horizon: int) -> np.ndarray:
  """Return quantiles shaped (series, horizon, level)."""
  width = max(len(c) for c in contexts)
  if any(len(c) != width for c in contexts):
    raise ValueError("upstream forecast() requires equal-length contexts")
  target = torch.tensor(np.stack(contexts)).unsqueeze(1)
  with torch.no_grad():
    quantiles = model.forecast(
      {
        "target": target,
        "target_mask": torch.ones_like(target, dtype=torch.bool),
        "series_ids": torch.zeros(len(contexts), 1, dtype=torch.long),
      },
      horizon=horizon,
      **FORECAST_KWARGS,
    )
  # (level, series, variate, horizon) -> (series, horizon, level)
  return quantiles.squeeze(2).permute(1, 2, 0).numpy()


def main() -> None:
  if len(sys.argv) != 2:
    raise SystemExit("usage: generate-toto-reference.py OUTPUT_DIR")
  source = Path(os.environ["TOTO_SOURCE"]).resolve()
  checkpoint = Path(os.environ["TOTO_CHECKPOINT"]).resolve()
  output_dir = Path(sys.argv[1]).resolve()
  assert_source_pin(source)
  sys.path.insert(0, str(source / "toto2"))

  import torch  # pylint: disable=import-outside-toplevel
  from safetensors.torch import load_file  # pylint: disable=import-outside-toplevel
  from toto2.configuration import Toto2ModelConfig  # pylint: disable=import-error,import-outside-toplevel
  from toto2.model import Toto2Model  # pylint: disable=import-error,import-outside-toplevel

  config = Toto2ModelConfig(**json.loads((checkpoint / "config.json").read_text()))
  model = Toto2Model(config)
  model.load_state_dict(load_file(checkpoint / "model.safetensors"))
  model.eval()
  output_dir.mkdir(parents=True, exist_ok=True)

  for case in fixture_cases():
    contexts = [materialize_context(spec) for spec in case["contexts"]]
    context_files = []
    for index, context in enumerate(contexts, start=1):
      context_file = f"{case['name']}-context-{index}.f32"
      np.asarray(context, dtype="<f4").tofile(output_dir / context_file)
      context_files.append(context_file)

    if case["name"] == "batch_agreement":
      # Unequal lengths are the point of this case, and upstream cannot batch
      # them, so each series is forecast alone. The R port batches them
      # together and must reproduce these same values.
      expected = np.stack([
        forecast(model, torch, [context], case["horizon"])[0]
        for context in contexts
      ])
    else:
      expected = forecast(model, torch, contexts, case["horizon"])

    payload = {
      "schema_version": 1,
      "name": case["name"],
      "source_commit": SOURCE_COMMIT,
      "model_id": CHECKPOINT_ID,
      "revision": CHECKPOINT_REVISION,
      "reference": {
        "numpy": np.__version__,
        "torch": torch.__version__,
        **FORECAST_KWARGS,
      },
      "context_specs": case["contexts"],
      "context_files": context_files,
      "patch_size": PATCH_SIZE,
      "horizon": case["horizon"],
      "quantile_levels": QUANTILE_LEVELS,
      "expected_quantiles": expected.tolist(),
      # See tests/testthat/fixtures/README.md: compared as
      # |actual - expected| <= atol + rtol * |expected|.
      "atol": 1e-4,
      "rtol": 1e-5,
    }
    (output_dir / f"{case['name']}.json").write_text(json.dumps(payload))
    print(f"wrote {case['name']}: {expected.shape}")


if __name__ == "__main__":
  main()
