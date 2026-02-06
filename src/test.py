"""Side-by-side test of SWH and Git backends."""

import difflib

EXCLUDE_EXTENSIONS = {
    ".pcap", ".png", ".jpg", ".jpeg", ".gif", ".ico", ".bmp",
    ".bin", ".exe", ".dll", ".so", ".o", ".a",
    ".zip", ".tar", ".gz", ".bz2", ".xz",
    ".pdf", ".doc", ".docx",
    ".pyc", ".class", ".jar",
}

from src.swh import SwhRepo

current = "swh:1:rev:d97e94223720684c6aa740ff219e0d19426c2220"

swh = SwhRepo()

parent_rev = swh.get_parent(current)

changed_files = swh.get_changed_files(current)

for path in changed_files:
    # skip path that is not text file.
    if any(path.endswith(ext) for ext in EXCLUDE_EXTENSIONS):
        print(f"Skipping binary file: {path}")
        continue

    current_content = swh.get_blob(current, path)
    parent_content = swh.get_blob(parent_rev, path)

    lines_parent = parent_content if parent_content is not None else []
    lines_current = current_content if current_content is not None else []
    
    result = []
    idx = 0
    matcher = difflib.SequenceMatcher(None, lines_parent, lines_current)

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for line in lines_parent[i1:i2]:
                result.append((idx, "", line))
                idx += 1
        elif tag == "delete":
            for line in lines_parent[i1:i2]:
                result.append((idx, "-", line))
                idx += 1
        elif tag == "insert":
            for line in lines_current[j1:j2]:
                result.append((idx, "+", line))
                idx += 1
        elif tag == "replace":
            for line in lines_parent[i1:i2]:
                result.append((idx, "-", line))
                idx += 1
            for line in lines_current[j1:j2]:
                result.append((idx, "+", line))
                idx += 1

    print(f"Diff for {path}:")
    for idx, tag, line in result:
        if tag == "+" or tag == "-":
            print(f"{idx:4d} {tag} {line}") 