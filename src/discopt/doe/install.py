"""Install the bundled Claude skill and expert agents.

Console script ``discopt-doe-install-skill``: copies the packaged ``SKILL.md``
into a Claude skills directory (auto-discovered by Claude Code) and the
bundled expert agents into the matching ``agents`` directory.
"""

from __future__ import annotations

import argparse
from importlib import resources
from pathlib import Path

SKILL_NAME = "discopt-doe"


def _skill_root():
    return resources.files("discopt.doe").joinpath("skill")


def install_skill(claude_dir: Path, force: bool = False) -> list[Path]:
    """Install SKILL.md and agents/*.md under ``claude_dir`` (a .claude dir)."""
    installed: list[Path] = []

    dest = Path(claude_dir) / "skills" / SKILL_NAME / "SKILL.md"
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and not force:
        print(f"skill already installed: {dest}\n  (use --force to overwrite)")
    else:
        dest.write_text(
            _skill_root().joinpath("SKILL.md").read_text(encoding="utf-8"), encoding="utf-8"
        )
        print(f"installed skill -> {dest}")
        installed.append(dest)

    agents_dir = Path(claude_dir) / "agents"
    agents_dir.mkdir(parents=True, exist_ok=True)
    for agent in _skill_root().joinpath("agents").iterdir():
        if not agent.name.endswith(".md"):
            continue
        adest = agents_dir / agent.name
        if adest.exists() and not force:
            print(f"agent already installed: {adest} (use --force to overwrite)")
            continue
        adest.write_text(agent.read_text(encoding="utf-8"), encoding="utf-8")
        print(f"installed agent -> {adest}")
        installed.append(adest)

    return installed


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        prog="discopt-doe-install-skill",
        description="Install the discopt-doe Claude skill and expert agents.",
    )
    scope = p.add_mutually_exclusive_group()
    scope.add_argument("--user", action="store_true", help="install to ~/.claude (default)")
    scope.add_argument(
        "--project", action="store_true", help="install to ./.claude (versioned with a repo)"
    )
    p.add_argument("--force", action="store_true", help="overwrite existing files")
    args = p.parse_args(argv)

    claude_dir = Path(".claude") if args.project else Path.home() / ".claude"
    install_skill(claude_dir, force=args.force)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
