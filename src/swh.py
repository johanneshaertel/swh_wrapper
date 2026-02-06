"""SWH archive backend for RepoAccess."""

import json
import os
from pathlib import Path
import requests

from dotenv import load_dotenv

load_dotenv()

SWH_API_BASE = "https://archive.softwareheritage.org/api/1"
SWH_TOKEN = os.getenv("SWH_TOKEN")
SWH_CACHE_DIR = Path("tmp/swh_cache")


def _swh_headers() -> dict:
    if SWH_TOKEN:
        return {"Authorization": f"Bearer {SWH_TOKEN}"}
    return {}


def _cache_path(category: str, key: str) -> Path:
    SWH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    safe_key = key.replace(":", "_").replace("/", "_")
    return SWH_CACHE_DIR / category / safe_key


def _cache_get(category: str, key: str) -> str | None:
    path = _cache_path(category, key)
    if path.exists():
        return path.read_text()
    return None


def _cache_set(category: str, key: str, value: str):
    path = _cache_path(category, key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def _swhid_to_hash(swhid: str) -> str:
    if swhid.startswith("swh:"):
        return swhid.split(":")[-1]
    return swhid


class SwhRepo:
    """SWH archive implementation of RepoAccess."""

    def _get_revision(self, swhid: str) -> dict:
        """Get revision metadata."""
        hash_id = _swhid_to_hash(swhid)

        cached = _cache_get("revision", hash_id)
        if cached:
            return json.loads(cached)

        url = f"{SWH_API_BASE}/revision/{hash_id}/"
        resp = requests.get(url, headers=_swh_headers())
        resp.raise_for_status()
        data = resp.json()

        _cache_set("revision", hash_id, json.dumps(data))
        return data

    def _get_directory_raw(self, dir_hash: str) -> list[dict]:
        """Get raw directory listing from SWH API."""
        cached = _cache_get("dir_list", dir_hash)
        if cached:
            return json.loads(cached)

        url = f"{SWH_API_BASE}/directory/{dir_hash}/"
        resp = requests.get(url, headers=_swh_headers())
        resp.raise_for_status()
        data = resp.json()

        _cache_set("dir_list", dir_hash, json.dumps(data))
        return data

    def _get_content_raw(self, content_hash: str) -> bytes:
        """Get raw content bytes."""
        cached = _cache_get("content", content_hash)
        if cached:
            return cached.encode()

        url = f"{SWH_API_BASE}/content/sha1_git:{content_hash}/raw/"
        resp = requests.get(url, headers=_swh_headers())
        resp.raise_for_status()
        data = resp.content

        _cache_set("content", content_hash, data.decode("utf-8", errors="replace"))
        return data

    def _resolve_path(self, dir_hash: str, path: str) -> tuple[str, str] | None:
        """Resolve path to (type, hash). Returns None if not found."""
        if not path:
            return ("dir", dir_hash)

        parts = path.split("/")
        current_hash = dir_hash

        for i, part in enumerate(parts):
            entries = self._get_directory_raw(current_hash)
            found = None
            for entry in entries:
                if entry["name"] == part:
                    found = entry
                    break

            if not found:
                return None

            if i == len(parts) - 1:
                entry_type = "file" if found["type"] == "file" else "dir"
                return (entry_type, found["target"])

            if found["type"] != "dir":
                return None

            current_hash = found["target"]

        return None

    def get_blob(self, revision: str, path: str) -> list[str] | None:
        rev = self._get_revision(revision)
        dir_hash = rev["directory"]

        resolved = self._resolve_path(dir_hash, path)
        if resolved is None or resolved[0] != "file":
            return None

        raw = self._get_content_raw(resolved[1])
        return raw.decode("utf-8", errors="replace").splitlines()

    def get_parent(self, revision: str) -> str | None:
        rev = self._get_revision(revision)
        parents = rev.get("parents", [])
        if not parents:
            return None
        return parents[0]["id"]

    def get_commit_message(self, revision: str) -> str:
        rev = self._get_revision(revision)
        return rev.get("message", "")

    def get_directory(self, revision: str, path: str = "") -> list[tuple[str, str]] | None:
        rev = self._get_revision(revision)
        dir_hash = rev["directory"]

        if path:
            resolved = self._resolve_path(dir_hash, path)
            if resolved is None or resolved[0] != "dir":
                return None
            dir_hash = resolved[1]

        entries = self._get_directory_raw(dir_hash)
        result = []
        for entry in entries:
            name = entry["name"]
            entry_type = "file" if entry["type"] == "file" else "dir"
            result.append((name, entry_type))

        return result

    def get_files(self, revision: str) -> list[str]:
        """Recursively list all files."""
        rev = self._get_revision(revision)
        dir_hash = rev["directory"]

        files = []
        self._collect_files(dir_hash, "", files)
        return files

    def _collect_files(self, dir_hash: str, prefix: str, files: list[str]):
        entries = self._get_directory_raw(dir_hash)
        for entry in entries:
            name = entry["name"]
            path = f"{prefix}{name}" if not prefix else f"{prefix}/{name}"

            if entry["type"] == "file":
                files.append(path)
            elif entry["type"] == "dir":
                self._collect_files(entry["target"], path, files)

    def get_changed_files(self, revision: str) -> list[str]:
        """Find changed files using Merkle tree diff."""
        parent_rev = self.get_parent(revision)
        if parent_rev is None:
            return self.get_files(revision)

        current = self._get_revision(revision)
        parent = self._get_revision(parent_rev)

        current_dir = current["directory"]
        parent_dir = parent["directory"]

        if current_dir == parent_dir:
            return []

        changed = []
        self._diff_trees(parent_dir, current_dir, "", changed)
        return changed

    def _diff_trees(self, parent_hash: str | None, current_hash: str | None, prefix: str, changed: list[str]):
        """Recursively diff two directory trees."""
        if parent_hash == current_hash:
            return

        parent_entries = {}
        current_entries = {}

        if parent_hash:
            for entry in self._get_directory_raw(parent_hash):
                parent_entries[entry["name"]] = (entry["type"], entry["target"])

        if current_hash:
            for entry in self._get_directory_raw(current_hash):
                current_entries[entry["name"]] = (entry["type"], entry["target"])

        all_names = set(parent_entries.keys()) | set(current_entries.keys())

        for name in all_names:
            path = name if not prefix else f"{prefix}/{name}"
            p_entry = parent_entries.get(name)
            c_entry = current_entries.get(name)

            if p_entry == c_entry:
                continue

            p_type, p_hash = p_entry if p_entry else (None, None)
            c_type, c_hash = c_entry if c_entry else (None, None)

            # File added, removed, or modified
            if c_type == "file" or p_type == "file":
                if p_type != c_type or p_hash != c_hash:
                    changed.append(path)

            # Recurse into directories with different hashes
            if c_type == "dir" or p_type == "dir":
                p_dir = p_hash if p_type == "dir" else None
                c_dir = c_hash if c_type == "dir" else None
                if p_dir != c_dir:
                    self._diff_trees(p_dir, c_dir, path, changed)

