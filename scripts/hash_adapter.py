"""
scripts/hash_adapter.py
Generate the SHA-256 hash for a LoRA adapter directory and, optionally,
write it into experts/manifest.json automatically.

Usage examples:
    # Print the hash only
    python scripts/hash_adapter.py experts/coder

    # Print the hash AND update manifest.json in one step
    python scripts/hash_adapter.py experts/coder --update-manifest

    # Verify an adapter already listed in the manifest
    python scripts/hash_adapter.py experts/coder --verify

IMPORTANT: This script uses the EXACT same hashing algorithm as
_hash_directory() in router.py.  Do not modify either in isolation —
any change to the algorithm invalidates every stored hash.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent.parent.resolve()
MANIFEST_PATH = PROJECT_ROOT / "experts" / "manifest.json"


# ---------------------------------------------------------------------------
# Core hashing logic — must stay identical to router._hash_directory()
# ---------------------------------------------------------------------------

def hash_directory(directory: Path) -> str:
    """
    Compute a deterministic SHA-256 over every file in *directory*.

    - Files are processed in sorted relative-path order (stable across OSes).
    - Both the relative file path and its binary contents are fed into the
      digest, so renaming a file changes the hash even if bytes are identical.
    - Directories themselves are not hashed; only regular files count.
    """
    if not directory.is_dir():
        print(f"[ERROR] Not a directory: {directory}", file=sys.stderr)
        sys.exit(1)

    sha = hashlib.sha256()
    files_hashed = 0

    for file_path in sorted(directory.rglob("*")):
        if file_path.is_file():
            rel = file_path.relative_to(directory).as_posix()
            sha.update(rel.encode())
            with file_path.open("rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    sha.update(chunk)
            files_hashed += 1

    if files_hashed == 0:
        print(
            f"[WARNING] Directory '{directory}' contains no files — "
            "hash covers an empty set.",
            file=sys.stderr,
        )

    return sha.hexdigest(), files_hashed


# ---------------------------------------------------------------------------
# Manifest helpers
# ---------------------------------------------------------------------------

def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        print(f"[ERROR] Manifest not found: {MANIFEST_PATH}", file=sys.stderr)
        sys.exit(1)
    with MANIFEST_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: dict) -> None:
    with MANIFEST_PATH.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"[OK] manifest.json updated: {MANIFEST_PATH}")


def find_expert_by_path(manifest: dict, adapter_path: Path) -> tuple[str, dict] | None:
    """Return (expert_name, entry) whose adapter_path resolves to *adapter_path*."""
    for name, entry in manifest.get("experts", {}).items():
        resolved = (PROJECT_ROOT / entry["adapter_path"]).resolve()
        if resolved == adapter_path.resolve():
            return name, entry
    return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hash a LoRA adapter directory for ExpertSwarm.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "adapter_path",
        help="Path to the adapter directory (relative to project root or absolute).",
    )
    group = p.add_mutually_exclusive_group()
    group.add_argument(
        "--update-manifest",
        action="store_true",
        help="Write the computed hash into experts/manifest.json.",
    )
    group.add_argument(
        "--verify",
        action="store_true",
        help="Compare the computed hash against the one stored in manifest.json.",
    )
    return p


def cmd_hash(adapter_dir: Path) -> str:
    """Compute and print the hash. Return the hex digest."""
    digest, n_files = hash_directory(adapter_dir)
    print(f"Adapter : {adapter_dir}")
    print(f"Files   : {n_files}")
    print(f"SHA-256 : {digest}")
    return digest


def cmd_update(adapter_dir: Path, digest: str) -> None:
    """Write *digest* into the matching manifest entry."""
    manifest = load_manifest()
    result = find_expert_by_path(manifest, adapter_dir)

    if result is None:
        print(
            f"[ERROR] No manifest entry found for '{adapter_dir}'.\n"
            "Add the expert to experts/manifest.json first, then re-run.",
            file=sys.stderr,
        )
        sys.exit(1)

    expert_name, entry = result
    old_hash = entry.get("sha256", "<none>")
    entry["sha256"] = digest
    save_manifest(manifest)
    print(f"Expert  : {expert_name}")
    print(f"Old hash: {old_hash}")
    print(f"New hash: {digest}")


def cmd_verify(adapter_dir: Path, digest: str) -> None:
    """Compare *digest* against the manifest and exit non-zero on mismatch."""
    import hmac
    manifest = load_manifest()
    result = find_expert_by_path(manifest, adapter_dir)

    if result is None:
        print(
            f"[ERROR] No manifest entry for '{adapter_dir}'.",
            file=sys.stderr,
        )
        sys.exit(1)

    expert_name, entry = result
    stored = entry.get("sha256", "")

    if stored.upper().startswith("REPLACE_WITH"):
        print(
            f"[FAIL] Expert '{expert_name}' still has a placeholder hash.\n"
            "Run with --update-manifest first.",
            file=sys.stderr,
        )
        sys.exit(1)

    if hmac.compare_digest(digest, stored.lower()):
        print(f"[OK] Expert '{expert_name}' — hash verified.")
    else:
        print(
            f"[FAIL] Hash mismatch for expert '{expert_name}'.\n"
            f"  Stored : {stored}\n"
            f"  Actual : {digest}",
            file=sys.stderr,
        )
        sys.exit(1)


def main() -> None:
    args = build_parser().parse_args()

    # Resolve the adapter path relative to project root if not absolute.
    adapter_dir = Path(args.adapter_path)
    if not adapter_dir.is_absolute():
        adapter_dir = PROJECT_ROOT / adapter_dir
    adapter_dir = adapter_dir.resolve()

    # Path-traversal guard: must stay within project root.
    try:
        adapter_dir.relative_to(PROJECT_ROOT)
    except ValueError:
        print(
            f"[ERROR] '{adapter_dir}' is outside the project root. Aborting.",
            file=sys.stderr,
        )
        sys.exit(1)

    digest = cmd_hash(adapter_dir)

    if args.update_manifest:
        print()
        cmd_update(adapter_dir, digest)
    elif args.verify:
        print()
        cmd_verify(adapter_dir, digest)


if __name__ == "__main__":
    main()
