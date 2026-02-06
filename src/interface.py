from typing import Protocol

class RepoAccess(Protocol):
    """Interface for accessing repository data."""

    def get_blob(self, revision: str, path: str) -> list[str] | None:
        """File content as lines at a revision. None if file doesn't exist."""
        ...

    def get_parent(self, revision: str) -> str | None:
        """Parent revision identifier. None if no parent (root commit)."""
        ...

    def get_commit_message(self, revision: str) -> str:
        """Commit message."""
        ...

    def get_changed_files(self, revision: str) -> list[str]:
        """Paths of files changed in this revision vs parent."""
        ...

    def get_files(self, revision: str) -> list[str]:
        """All file paths in the repository at this revision."""
        ...

    def get_directory(self, revision: str, path: str = "") -> list[tuple[str, str]] | None:
        """Directory entries at path. Returns list of (name, type) or None.

        type is 'file' or 'dir'.
        """
        ...
