"""Local git repository backend for RepoAccess."""

import subprocess
from pathlib import Path


class GitRepo:
    """Local bare git repository implementation of RepoAccess."""

    def __init__(self, repo_path: Path | str):
        self.repo_path = Path(repo_path)
        assert self.repo_path.exists(), f"Repository not found: {repo_path}"

    def _run(self, *args, check: bool = True) -> subprocess.CompletedProcess:
        return subprocess.run(
            ["git", "-C", str(self.repo_path)] + list(args),
            capture_output=True,
            text=True,
            check=check,
        )

    def get_blob(self, revision: str, path: str) -> list[str] | None:
        result = self._run("show", f"{revision}:{path}", check=False)
        if result.returncode != 0:
            return None
        return result.stdout.splitlines()

    def get_parent(self, revision: str) -> str | None:
        result = self._run("rev-parse", f"{revision}^", check=False)
        if result.returncode != 0:
            return None
        return result.stdout.strip()

    def get_commit_message(self, revision: str) -> str:
        result = self._run("log", "-1", "--format=%s%n%n%b", revision)
        return result.stdout.strip()

    def get_directory(self, revision: str, path: str = "") -> list[tuple[str, str]] | None:
        if path and not path.endswith("/"):
            path = path + "/"

        args = ["ls-tree", revision]
        if path:
            args.append(path)

        result = self._run(*args, check=False)
        if result.returncode != 0:
            return None

        entries = []
        for line in result.stdout.strip().split("\n"):
            if not line:
                continue
            # Format: <mode> <type> <hash>\t<name>
            meta, name = line.split("\t", 1)
            parts = meta.split()
            obj_type = "file" if parts[1] == "blob" else "dir"
            obj_hash = parts[2]
            # Strip path prefix if present
            if path and name.startswith(path):
                name = name[len(path):]
            entries.append((name, obj_type))

        return entries

    def get_files(self, revision: str) -> list[str]:
        result = self._run("ls-tree", "-r", "--name-only", revision)
        return [line for line in result.stdout.strip().split("\n") if line]

    def get_changed_files(self, revision: str) -> list[str]:
        parent = self.get_parent(revision)
        if parent is None:
            return self.get_files(revision)

        result = self._run("diff-tree", "-r", "--name-only", "--no-commit-id", revision)
        return [line for line in result.stdout.strip().split("\n") if line]
