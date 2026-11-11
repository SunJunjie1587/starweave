from __future__ import annotations

import pytest

from starweave.cli.main import main

pytestmark = pytest.mark.npu


def test_doctor_sees_npu() -> None:
    pytest.importorskip("torch")
    pytest.importorskip("torch_npu")
    assert main(["doctor"]) == 0
