"""Record the reference outputs: ``uv run python -m test.integration.reference.generate``."""

from __future__ import annotations

from pathlib import Path

import jax
import numpy as onp

jax.config.update("jax_enable_x64", True)

from test.integration.reference.scenarios import run_all  # noqa: E402

PATH = Path(__file__).with_name("reference_outputs.npz")


def main() -> None:
    outputs = {key: onp.asarray(value) for key, value in run_all().items()}
    onp.savez_compressed(PATH, **outputs)
    print(f"wrote {len(outputs)} arrays to {PATH}")


if __name__ == "__main__":
    main()
