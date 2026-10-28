"""Channel estimators registered from a plugin file outside the package."""

import pytest

from nr_ul_sim import parameters
from nr_ul_sim.channel_estimation import build_estimator
from nr_ul_sim.cli import build_parser, config_from_args
from nr_ul_sim.parameters import SimConfig, available_estimators, register_estimator

PLUGIN = '''
from nr_ul_sim.parameters import register_estimator
from helper import MARKER

register_estimator("toy-ce", lambda transmitter, cfg: (MARKER, transmitter, cfg), "toy")
'''


@pytest.fixture
def plugin_file(tmp_path, monkeypatch):
    monkeypatch.setattr(parameters, "PLUGIN_ESTIMATORS", dict(parameters.PLUGIN_ESTIMATORS))
    monkeypatch.setattr(parameters, "_LOADED_PLUGINS", dict(parameters._LOADED_PLUGINS))
    monkeypatch.setattr(parameters, "ESTIMATOR_INFO", dict(parameters.ESTIMATOR_INFO))
    import nr_ul_sim.channel_estimation as ce
    monkeypatch.setattr(ce, "PLUGIN_ESTIMATORS", parameters.PLUGIN_ESTIMATORS)
    # the plugin imports a sibling module from its own directory
    (tmp_path / "helper.py").write_text("MARKER = 'toy'\n")
    path = tmp_path / "plugin.py"
    path.write_text(PLUGIN)
    return path


def test_simconfig_loads_plugin(plugin_file):
    assert "toy_ce" not in available_estimators()
    cfg = SimConfig(channel_estimators=("ls_lin", "toy-ce"), ce_plugins=(str(plugin_file),))
    assert cfg.channel_estimators == ("ls_lin", "toy_ce")
    assert "toy_ce" in available_estimators()
    assert cfg.summary()["ce_plugins"] == [str(plugin_file)]
    marker, tx, got = build_estimator("toy_ce", "tx", cfg)
    assert (marker, tx, got) == ("toy", "tx", cfg)


def test_cli_plugin(plugin_file):
    args = build_parser().parse_args(
        ["--estimators", "ls_lin,toy_ce", "--ce-plugin", str(plugin_file)])
    cfg = config_from_args(args)
    assert cfg.channel_estimators == ("ls_lin", "toy_ce")


def test_unknown_without_plugin(plugin_file):
    with pytest.raises(ValueError, match="toy_ce"):
        SimConfig(channel_estimators=("toy_ce",))


def test_builtin_names_are_protected(plugin_file):
    for name in ("ls_lin", "lmmse", "perfect"):
        with pytest.raises(ValueError, match="built in"):
            register_estimator(name, lambda tx, cfg: None)


def test_missing_plugin_file(tmp_path):
    with pytest.raises(ValueError, match="not a file"):
        SimConfig(ce_plugins=(str(tmp_path / "missing.py"),))
