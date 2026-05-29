from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Iterable, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download and prepare MURA-v1.1 into ImageFolder (binary).")
    parser.add_argument("--dest", type=Path, required=True, help="Destination root directory for MURA.")
    parser.add_argument(
        "--method",
        type=str,
        choices=["url", "kaggle", "zip"],
        default="url",
        help="Download method: url (Google storage), kaggle (requires kaggle CLI), or zip (use local zip).",
    )
    parser.add_argument(
        "--zip-path",
        type=Path,
        default=None,
        help="Path to MURA-v1.1.zip if method=zip.",
    )
    parser.add_argument(
        "--kaggle-dataset",
        type=str,
        default="paultimothymooney/mura-v11",
        help="Kaggle dataset slug (method=kaggle).",
    )
    parser.add_argument(
        "--url",
        type=str,
        default="https://storage.googleapis.com/mura_dataset/MURA-v1.1.zip",
        help="Direct URL to MURA-v1.1.zip (method=url).",
    )
    parser.add_argument(
        "--link",
        action="store_true",
        help="Use hardlinks instead of copying when building ImageFolder (saves space).",
    )
    return parser.parse_args()


def ensure_dirs(dest_root: Path) -> Tuple[Path, Path]:
    raw_dir = dest_root / "raw"
    out_dir = dest_root / "imagefolder"
    raw_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    return raw_dir, out_dir


def download_via_url(url: str, target_zip: Path) -> None:
    try:
        import requests  # type: ignore
    except Exception as exc:
        raise SystemExit("requests is required for --method url (pip install requests)") from exc
    target_zip.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with target_zip.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)


def download_via_kaggle(dataset: str, raw_dir: Path) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    # Requires kaggle CLI configured with API token
    cmd = ["kaggle", "datasets", "download", "-d", dataset, "-p", str(raw_dir), "-q"]
    subprocess.run(cmd, check=True)
    # Find the downloaded zip
    zips = list(raw_dir.glob("*.zip"))
    if not zips:
        raise SystemExit("Kaggle download finished but no zip file found.")
    return zips[0]


def extract_zip(zip_path: Path, extract_to: Path) -> Path:
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    # Return directory that contains 'MURA-v1.1'
    cand = extract_to / "MURA-v1.1"
    if cand.exists():
        return cand
    # Some zips might nest differently; fallback to find
    for p in extract_to.iterdir():
        if p.is_dir() and p.name.lower().startswith("mura"):
            return p
    raise SystemExit("Could not locate extracted MURA root.")


def iter_images(root: Path) -> Iterable[Path]:
    exts = {".png", ".jpg", ".jpeg"}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            yield p


def build_imagefolder(src_split_dir: Path, dst_split_dir: Path, use_link: bool) -> None:
    # Create NEGATIVE / POSITIVE folders
    neg_dir = dst_split_dir / "NEGATIVE"
    pos_dir = dst_split_dir / "POSITIVE"
    neg_dir.mkdir(parents=True, exist_ok=True)
    pos_dir.mkdir(parents=True, exist_ok=True)
    # Copy/link images based on path keyword
    count_pos = 0
    count_neg = 0
    for img in iter_images(src_split_dir):
        # MURA names studies like *_positive or *_negative in path
        path_str = str(img).lower()
        is_pos = ("positive" in path_str) and ("negative" not in path_str)
        dst_dir = pos_dir if is_pos else neg_dir
        dst_path = dst_dir / f"{img.stem}{img.suffix.lower()}"
        # avoid overwrite
        suffix = 1
        while dst_path.exists():
            dst_path = dst_dir / f"{img.stem}_{suffix:02d}{img.suffix.lower()}"
            suffix += 1
        if use_link:
            try:
                os.link(img, dst_path)
            except OSError:
                shutil.copy2(img, dst_path)
        else:
            shutil.copy2(img, dst_path)
        if is_pos:
            count_pos += 1
        else:
            count_neg += 1
    print(f"[{src_split_dir.name}] POSITIVE: {count_pos}  NEGATIVE: {count_neg}")


def main() -> None:
    args = parse_args()
    raw_dir, out_dir = ensure_dirs(args.dest)
    # Step 1: obtain zip
    if args.method == "url":
        zip_path = raw_dir / "MURA-v1.1.zip"
        if not zip_path.exists():
            print(f"[INFO] Downloading from URL to {zip_path} ...")
            download_via_url(args.url, zip_path)
        else:
            print(f"[INFO] Reusing existing zip: {zip_path}")
    elif args.method == "kaggle":
        print(f"[INFO] Downloading via Kaggle: {args.kaggle_dataset}")
        zip_path = download_via_kaggle(args.kaggle_dataset, raw_dir)
        print(f"[INFO] Downloaded: {zip_path}")
    else:
        if not args.zip_path or not args.zip_path.exists():
            raise SystemExit("--zip requires a valid --zip-path to MURA-v1.1.zip")
        zip_path = args.zip_path
        print(f"[INFO] Using local zip: {zip_path}")

    # Step 2: extract
    extracted_root = raw_dir / "extracted"
    mura_root = extract_zip(zip_path, extracted_root)
    print(f"[INFO] Extracted MURA root: {mura_root}")
    # Expect 'train' and 'valid' folders under mura_root
    train_src = mura_root / "train"
    valid_src = mura_root / "valid"
    if not train_src.exists() or not valid_src.exists():
        raise SystemExit("Extracted MURA does not contain expected 'train' and 'valid' directories.")

    # Step 3: build ImageFolder binary layout
    train_dst = out_dir / "train"
    val_dst = out_dir / "val"
    # reset targets
    if train_dst.exists():
        shutil.rmtree(train_dst)
    if val_dst.exists():
        shutil.rmtree(val_dst)
    train_dst.mkdir(parents=True, exist_ok=True)
    val_dst.mkdir(parents=True, exist_ok=True)

    print("[INFO] Building ImageFolder(train)...")
    build_imagefolder(train_src, train_dst, args.link)
    print("[INFO] Building ImageFolder(val)...")
    build_imagefolder(valid_src, val_dst, args.link)
    print(f"[DONE] ImageFolder prepared at: {out_dir}")
    print(f"       Train: {train_dst}")
    print(f"       Val  : {val_dst}")


if __name__ == "__main__":
    main()


