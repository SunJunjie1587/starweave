from __future__ import annotations

import subprocess
import sys

import pytest
from pydantic import ValidationError

from starweave.seams import (
    DataConfig,
    OptimizerConfig,
    ParallelPlan,
    Rebindable,
    TrainerConfig,
)


def test_trainer_defaults() -> None:
    config = TrainerConfig()
    assert config.max_steps == 1000
    assert config.micro_batch_size == 1


def test_data_config_rejects_extra() -> None:
    with pytest.raises(ValidationError):
        DataConfig(path="x", nope=True)


def test_optimizer_betas_tuple() -> None:
    config = OptimizerConfig(betas=[0.9, 0.95])
    assert config.betas == (0.9, 0.95)


def test_parallel_plan_defaults() -> None:
    plan = ParallelPlan()
    assert plan.shard_modules == ()
    assert plan.expert_modules == ()


def test_seams_do_not_import_torch() -> None:
    code = (
        "import sys; import starweave.seams; "
        "assert 'torch' not in sys.modules, 'seams imported torch'"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


def test_rebindable_protocol_shape() -> None:
    class Impl:
        def flush(self) -> None: ...
        def close(self) -> None: ...
        def release(self) -> None: ...

    assert isinstance(Impl(), Rebindable)
