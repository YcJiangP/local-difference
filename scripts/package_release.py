"""Build a source archive with approved documentation assets and no participant data."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
TOP_FILES = {
    "README.md", "LICENSE", "NOTICE.md", "pyproject.toml", "requirements.txt",
    ".gitignore", ".gitattributes",
}
TOP_DIRECTORIES = {"src", "docs", "tests", "examples", "scripts", ".github", "assets"}
DOCUMENTATION_ASSETS = {"assets/table_ii.png"}
FORBIDDEN_SUFFIXES = {
    ".wav", ".flac", ".mp3", ".pt", ".pth", ".bin", ".safetensors",
    ".npz", ".npy", ".joblib", ".pkl", ".pickle", ".pdf",
}


def source_files():
    files = []
    for path in sorted(ROOT.rglob("*")):
        relative = path.relative_to(ROOT)
        if any(part in {"__pycache__", ".git"} or part.endswith(".egg-info") for part in relative.parts):
            continue
        if not path.is_file():
            continue
        if relative.as_posix() not in TOP_FILES and relative.parts[0] not in TOP_DIRECTORIES:
            continue
        if path.is_symlink() or not path.resolve().is_relative_to(ROOT.resolve()):
            raise ValueError(f"External or symbolic-link target: {relative}")
        if path.suffix.lower() in FORBIDDEN_SUFFIXES:
            raise ValueError(f"Non-source artifact in release tree: {relative}")
        if relative.parts[0] == "assets":
            if relative.as_posix() not in DOCUMENTATION_ASSETS:
                raise ValueError(f"Unapproved documentation asset: {relative}")
            payload = path.read_bytes()
            if not payload.startswith(b"\x89PNG\r\n\x1a\n") or len(payload) > 5_000_000:
                raise ValueError(f"Invalid documentation PNG: {relative}")
            files.append(path)
            continue
        content = path.read_text(encoding="utf-8")
        for expression, reason in (
            (r"PROCESS-2_rec__\d+", "clinical participant identifier"),
            (r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]", "absolute machine path"),
            (r"-----BEGIN [A-Z ]*PRIVATE KEY-----", "private key"),
            (r"hf_[A-Za-z0-9]{20,}", "access token"),
            (r"gh[pousr]_[A-Za-z0-9]{20,}", "access token"),
            (r"[\u3400-\u9fff]", "non-English CJK content"),
        ):
            if re.search(expression, content):
                raise ValueError(f"Unexpected {reason} in {relative}")
        if path.suffix == ".py":
            ast.parse(content)
        files.append(path)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.is_relative_to(ROOT.resolve()) or output.suffix.lower() != ".zip":
        raise ValueError("Write a .zip archive outside the repository directory.")
    files = source_files()
    required = [ROOT / name for name in TOP_FILES | DOCUMENTATION_ASSETS]
    if not all(path in files for path in required):
        raise ValueError("Missing required release metadata.")
    manifest = {
        "format": 1, "package": "local-temporal-speech-ci", "version": "0.1.0",
        "license": "MIT", "source_only": True,
        "documentation_assets": sorted(DOCUMENTATION_ASSETS),
        "files": [
            {
                "path": path.relative_to(ROOT).as_posix(), "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in files
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
        for path in files:
            archive.write(path, ROOT.name + "/" + path.relative_to(ROOT).as_posix())
        archive.writestr(ROOT.name + "/RELEASE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({
        "archive": str(output), "source_files": len(files), "bytes": output.stat().st_size,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }, indent=2))


if __name__ == "__main__":
    main()
