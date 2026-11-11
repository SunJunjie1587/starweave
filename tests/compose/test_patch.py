from __future__ import annotations

import pytest

from starweave.compose.patch import Composer
from starweave.kernel.errors import ComposeError


def test_insert_and_override_replaces_config_wholesale() -> None:
    composer = Composer()
    composer.apply(
        [{"insert": [{"id": "trainer", "plugin": "trainer-sft", "config": {"max_steps": 10, "log_every": 5}}]}],
        "bundle:test",
    )
    composer.apply([{"id": "trainer", "config": {"log_every": 20}}], "recipe.patch")
    row = composer.rows[0]
    assert row.config == {"log_every": 20}


def test_duplicate_id_rejected() -> None:
    composer = Composer()
    composer.apply([{"insert": [{"id": "x", "plugin": "metrics-noop"}]}], "bundle:test")
    with pytest.raises(ComposeError):
        composer.apply([{"insert": [{"id": "x", "plugin": "metrics-noop"}]}], "bundle:test")


def test_unknown_target_rejected() -> None:
    composer = Composer()
    with pytest.raises(ComposeError):
        composer.apply([{"id": "missing", "config": {}}], "recipe.patch")


def test_inject_reserved() -> None:
    composer = Composer()
    with pytest.raises(ComposeError):
        composer.apply([{"insert": [{"id": "x", "plugin": "metrics-noop", "inject": ["a"]}]}], "bundle:test")


def test_group_expands_and_accepts_insert() -> None:
    composer = Composer()
    composer.apply(
        [
            {
                "insert": [
                    {
                        "id": "policies",
                        "group": True,
                        "config": [
                            {"id": "policy-a", "plugin": "policy-grad-clip"},
                        ],
                    }
                ]
            }
        ],
        "bundle:test",
    )
    composer.apply(
        [{"insert": [{"id": "policy-b", "plugin": "policy-nan-guard"}], "id": "policies"}],
        "recipe.patch",
    )
    ids = {row.id for row in composer.rows}
    assert ids == {"policy-a", "policy-b"}
    assert len(composer.groups["policies"]) == 2
