"""Packaging-level tests: entry point, namespace resolution, package data."""

import importlib.metadata
from importlib import resources

import pytest

pytestmark = pytest.mark.smoke


def test_cli_entry_point_resolves():
    eps = {ep.name: ep for ep in importlib.metadata.entry_points(group="discopt.cli")}
    assert "doe" in eps, "discopt-doe must register the 'doe' subcommand"
    mod = eps["doe"].load()
    assert callable(mod.add_subparser)
    assert callable(mod.run)


def test_namespace_import():
    import discopt
    import discopt.doe

    # The plugin merges into the discopt namespace; both must be importable
    # side by side, and the public API surface must be present.
    assert hasattr(discopt.doe, "compute_fim")
    assert hasattr(discopt.doe, "optimal_experiment")


def test_gui_logo_ships_as_package_data():
    logo = resources.files("discopt.doe").joinpath("gui", "discopt-logo.png")
    assert logo.is_file()


def test_skill_ships_as_package_data():
    skill = resources.files("discopt.doe").joinpath("skill", "SKILL.md")
    assert skill.is_file()
    text = skill.read_text()
    assert text.startswith("---")
    assert "name: discopt-doe" in text


def test_version_prefix_tolerates_prereleases():
    """The discopt version guard must accept pre-releases like 0.6rc1."""
    from discopt.doe import _version_prefix

    assert _version_prefix("0.6") == (0, 6)
    assert _version_prefix("0.6rc1") == (0, 6)
    assert _version_prefix("0.7b2") == (0, 7)
    assert _version_prefix("0.6.0.dev0") == (0, 6)
    assert _version_prefix("0.10.1") == (0, 10)
    assert _version_prefix("0.5") == (0, 5)
    assert _version_prefix("0.6rc1") >= (0, 6)
    assert _version_prefix("0.5") < (0, 6)
