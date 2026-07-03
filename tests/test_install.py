"""Tests for the skill installer (discopt-doe-install-skill)."""

import pytest

from discopt.doe.install import SKILL_NAME, install_skill, main

pytestmark = pytest.mark.smoke

EXPECTED_AGENTS = {
    "doe-expert.md",
    "estimability-expert.md",
    "identifiability-expert.md",
    "model-discrimination-expert.md",
}


def test_install_writes_skill_and_agents(tmp_path):
    installed = install_skill(tmp_path)
    skill = tmp_path / "skills" / SKILL_NAME / "SKILL.md"
    assert skill.exists()
    assert skill.read_text().startswith("---")
    agent_names = {p.name for p in (tmp_path / "agents").glob("*.md")}
    assert agent_names == EXPECTED_AGENTS
    assert len(installed) == 1 + len(EXPECTED_AGENTS)


def test_install_is_idempotent_without_force(tmp_path, capsys):
    install_skill(tmp_path)
    skill = tmp_path / "skills" / SKILL_NAME / "SKILL.md"
    skill.write_text("locally edited")
    installed = install_skill(tmp_path)  # no force: must not overwrite
    assert installed == []
    assert skill.read_text() == "locally edited"
    assert "already installed" in capsys.readouterr().out


def test_force_overwrites(tmp_path):
    install_skill(tmp_path)
    skill = tmp_path / "skills" / SKILL_NAME / "SKILL.md"
    skill.write_text("locally edited")
    install_skill(tmp_path, force=True)
    assert skill.read_text().startswith("---")


def test_main_project_scope(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert main(["--project"]) == 0
    assert (tmp_path / ".claude" / "skills" / SKILL_NAME / "SKILL.md").exists()
    assert (tmp_path / ".claude" / "agents" / "doe-expert.md").exists()
