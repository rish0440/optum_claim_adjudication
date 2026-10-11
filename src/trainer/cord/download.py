"""Manual, six-file Hub download with revision pinning and integrity checks."""
import argparse
import gzip
import json
from pathlib import Path
import re
import shutil
import time
import zipfile

from src.utils.paths import project_path
from .schema import DEFAULT_REVISION, FILES, REPO_ID, sha, write_json


def verify_source(source):
    source = Path(source).resolve()
    manifest = json.loads((source / "source_manifest.json").read_text())
    if manifest.get("repo_id") != REPO_ID or not re.fullmatch(r"[a-f0-9]{40}", manifest.get("revision", "")):
        raise ValueError("Invalid CORD source identity/revision")
    if set(manifest.get("files", {})) != set(FILES):
        raise ValueError("Source manifest must identify exactly six files")
    for name in FILES:
        path, info = source / name, manifest["files"][name]
        if not path.resolve().is_relative_to(source) or not path.is_file():
            raise ValueError(f"Missing or unsafe source file: {name}")
        if path.stat().st_size != info["size"] or sha(path) != info["sha256"]:
            raise ValueError(f"Source checksum/size mismatch: {name}")
    return manifest


def download(source, revision, retries=3):
    # No imports or network activity occur until this manual command is called.
    from huggingface_hub import HfApi, hf_hub_download
    from huggingface_hub.utils import enable_progress_bars
    if retries < 1:
        raise ValueError("Retries must be positive")
    enable_progress_bars()
    info = HfApi(token=False).dataset_info(REPO_ID, revision=revision, files_metadata=True, timeout=60)
    if not re.fullmatch(r"[a-f0-9]{40}", info.sha or ""):
        raise ValueError("Hub did not resolve a full commit SHA")
    siblings = {file.rfilename: file for file in info.siblings}
    metadata = {}
    for name in FILES:
        file = siblings.get(name)
        if file is None or file.lfs is None or file.size is None:
            raise ValueError(f"Missing file or cryptographic Hub metadata: {name}")
        # BlobLfsInfo is an attribute object in supported huggingface_hub releases.
        digest = file.lfs.sha256
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise ValueError(f"Missing Hub SHA256: {name}")
        metadata[name] = {"size": file.size, "sha256": digest,
                          "url": f"https://huggingface.co/datasets/{REPO_ID}/resolve/{info.sha}/{name}"}
    source = Path(source).resolve()
    source.mkdir(parents=True, exist_ok=True)
    if not (source / ".cache").resolve().is_relative_to(source):
        raise ValueError("Source cache path escapes the download directory")
    manifest = {"repo_id": REPO_ID, "requested_revision": revision, "revision": info.sha, "files": metadata}
    manifest_path = source / "source_manifest.json"
    if manifest_path.exists():
        old = json.loads(manifest_path.read_text())
        if old.get("revision") != info.sha or old.get("files") != metadata:
            raise ValueError("Source directory belongs to another revision; choose a new --source-dir")
    else:
        write_json(manifest_path, manifest)
    needed = sum(value["size"] for name, value in metadata.items() if not (source / name).exists())
    if shutil.disk_usage(source).free < 2 * needed + 512 * 1024 * 1024:
        raise ValueError("Insufficient disk space: reserve twice remaining download bytes plus 512 MiB")
    print(f"CORD revision {info.sha}; source {source}; remaining {needed:,} bytes", flush=True)
    for index, name in enumerate(FILES, 1):
        target = source / name
        print(f"[{index}/6] {name}", flush=True)
        if target.is_symlink():
            raise ValueError(f"Refusing a symlinked source download: {name}")
        if target.exists():
            if not target.resolve().is_relative_to(source) or sha(target) != metadata[name]["sha256"]:
                raise ValueError(f"Existing source differs; refusing overwrite: {name}")
            print("  verified existing file; reused", flush=True)
            continue
        for attempt in range(retries):
            try:
                downloaded = Path(hf_hub_download(repo_id=REPO_ID, repo_type="dataset", filename=name,
                    revision=info.sha, local_dir=source, cache_dir=source / ".cache/hub", token=False))
                if downloaded.resolve() != target.resolve():
                    raise ValueError("Unexpected Hub download destination")
                break
            except Exception as exc:
                if attempt + 1 == retries:
                    raise RuntimeError(f"Download failed for {name} ({type(exc).__name__}); rerun to resume") from exc
                time.sleep(min(2 ** attempt, 8))
        if target.stat().st_size != metadata[name]["size"] or sha(target) != metadata[name]["sha256"]:
            raise ValueError(f"Downloaded integrity failure: {name}; no automatic deletion")
        if name.endswith(".zip"):
            with zipfile.ZipFile(target) as archive:
                if not archive.infolist():
                    raise ValueError(f"Empty ZIP: {name}")
        else:
            with gzip.open(target, "rb") as stream:
                while stream.read(1024 * 1024):
                    pass  # Validate the gzip footer/CRC without extracting files.
    verify_source(source)
    print("All six source files verified; no receipt images extracted", flush=True)


def cleanup(source, dataset):
    """Explicit user-invoked deletion of our three source ZIPs only, after validation."""
    from .validate import validate_dataset
    source, dataset = Path(source).resolve(), Path(dataset).resolve()
    manifest = verify_source(source)
    validate_dataset(dataset)
    selected = json.loads((dataset / "manifest.json").read_text())
    if selected["source"] != manifest:
        raise ValueError("Prepared dataset does not belong to this source manifest")
    for name in FILES:
        if name.endswith(".zip"):
            (source / name).unlink()
            print(f"Removed verified temporary archive: {name}", flush=True)
    print("Annotations, source manifest and all caches retained. Source preparation now requires re-download of ZIPs.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, default=Path("data/cord_source"))
    parser.add_argument("--revision", default=DEFAULT_REVISION)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--cleanup-archives", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=Path("data/cord_200"))
    args = parser.parse_args()
    if args.cleanup_archives:
        cleanup(project_path(args.source_dir), project_path(args.dataset_root))
    else:
        download(project_path(args.source_dir), args.revision, args.retries)


if __name__ == "__main__":
    main()
