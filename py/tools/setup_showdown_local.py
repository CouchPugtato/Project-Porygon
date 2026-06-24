from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


DEFAULT_SERVER_REPO_URL = "https://github.com/smogon/pokemon-showdown.git"
DEFAULT_SERVER_TARGET_DIR = Path("external/pokemon-showdown")
DEFAULT_CLIENT_REPO_URL = "https://github.com/smogon/pokemon-showdown-client.git"
DEFAULT_CLIENT_TARGET_DIR = Path("external/pokemon-showdown-client")


def run(command: list[str], cwd: Path | None = None) -> None:
    print(f"[setup_showdown_local] $ {' '.join(command)}")
    subprocess.run(command, cwd=str(cwd) if cwd is not None else None, check=True)


def require_command(name: str) -> None:
    if shutil.which(name) is None:
        raise SystemExit(f"required command not found on PATH: {name}")


def npm_command() -> str:
    if os.name == "nt" and shutil.which("npm.cmd"):
        return "npm.cmd"
    return "npm"


def patch_client_build_for_esm(target_dir: Path) -> None:
    update_path = target_dir / "build-tools" / "update"
    if not update_path.exists():
        return
    source = update_path.read_text(encoding="utf-8")
    marker = "/* project-porygon local compatibility patch */"
    needle = "const compiler = require('./compiler.mjs');"
    if marker in source or needle not in source:
        return
    replacement = (
        f"let compiler;\n{marker}\n(async () => {{\n"
        "compiler = await import('./compiler.mjs');"
    )
    patched = source.replace(needle, replacement, 1)
    patched += "\n})().catch(err => {\n\tconsole.error(err);\n\tprocess.exit(1);\n});\n"
    update_path.write_text(patched, encoding="utf-8")
    print(f"[setup_showdown_local] patched client build script for ESM compatibility: {update_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-repo-url", default=DEFAULT_SERVER_REPO_URL)
    parser.add_argument("--server-target-dir", default=str(DEFAULT_SERVER_TARGET_DIR))
    parser.add_argument("--client-repo-url", default=DEFAULT_CLIENT_REPO_URL)
    parser.add_argument("--client-target-dir", default=str(DEFAULT_CLIENT_TARGET_DIR))
    parser.add_argument("--skip-server", action="store_true")
    parser.add_argument("--skip-client", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--skip-client-build", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser


def ensure_repo(repo_root: Path, repo_url: str, target_dir_value: str, skip_install: bool, skip_build: bool, force: bool, build_client: bool) -> None:
    target_dir = (repo_root / target_dir_value).resolve()
    target_parent = target_dir.parent

    if target_dir.exists():
        if not force:
            raise SystemExit(
                f"target directory already exists: {target_dir}\n"
                f"use --force to reuse it and still run install/build steps"
            )
        print(f"[setup_showdown_local] reusing existing directory: {target_dir}")
    else:
        target_parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", repo_url, str(target_dir)])

    if not skip_install:
        run([npm_command(), "install"], cwd=target_dir)

    if build_client and not skip_build:
        patch_client_build_for_esm(target_dir)
        try:
            run(["node", "build"], cwd=target_dir)
        except subprocess.CalledProcessError:
            print(
                "[setup_showdown_local] warning: local client build failed; "
                "continuing because testclient-old.html may still be usable from the cloned repo"
            )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    repo_root = Path.cwd()

    require_command("git")
    require_command("node")
    if not args.skip_install:
        require_command(npm_command())

    if not args.skip_server:
        ensure_repo(
            repo_root,
            args.server_repo_url,
            args.server_target_dir,
            skip_install=args.skip_install,
            skip_build=True,
            force=args.force,
            build_client=False,
        )

    if not args.skip_client:
        ensure_repo(
            repo_root,
            args.client_repo_url,
            args.client_target_dir,
            skip_install=args.skip_install,
            skip_build=args.skip_client_build,
            force=args.force,
            build_client=True,
        )

    print("[setup_showdown_local] ready")
    if not args.skip_server:
        print(f"[setup_showdown_local] server dir: {(repo_root / args.server_target_dir).resolve()}")
    if not args.skip_client:
        print(f"[setup_showdown_local] client dir: {(repo_root / args.client_target_dir).resolve()}")


if __name__ == "__main__":
    main()
