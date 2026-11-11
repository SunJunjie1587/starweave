from __future__ import annotations

import unittest

from example_adamw.config import AdamWConfig
from example_adamw.plugin import AdamWPlugin


class FakeContext:
    """Minimal context: the plugin must not need the real kernel."""

    def __init__(self) -> None:
        self.effects = []
        self.events = []

    def effect(self, fn):
        self.effects.append(fn)

    def emit(self, name, payload=None):
        self.events.append((name, payload))


class ContractTest(unittest.TestCase):
    def test_metadata(self) -> None:
        self.assertEqual(AdamWPlugin.name, "adamw")
        self.assertEqual(AdamWPlugin.provides, ("optimizer",))
        self.assertEqual(AdamWPlugin.requires, ("device", "model"))
        self.assertIs(AdamWPlugin.config_model, AdamWConfig)

    def test_config_defaults(self) -> None:
        config = AdamWConfig()
        self.assertGreater(config.lr, 0)
        self.assertGreaterEqual(config.weight_decay, 0)

    def test_config_rejects_unknown_key(self) -> None:
        with self.assertRaises(Exception):
            AdamWConfig(nope=True)

    def test_setup_registers_teardown(self) -> None:
        ctx = FakeContext()
        plugin = AdamWPlugin()
        plugin.setup(ctx, AdamWConfig())
        self.assertIn(plugin.teardown, ctx.effects)
        plugin.teardown()
        plugin.teardown()


if __name__ == "__main__":
    unittest.main()
