from __future__ import annotations
from pathlib import Path
import random
import re
import shutil

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

def normalize_label(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", "_", s)
    s = re.sub(r"[^a-z0-9_]+", "", s)
    return s

def safe_filename(name: str, max_len: int = 120) -> str:
    base = Path(name).stem
    ext = Path(name).suffix.lower()

    base = re.sub(r'[<>:"/\\|?*]+', "_", base)
    base = re.sub(r"\s+", "_", base)
    base = re.sub(r"_+", "_", base)

    if len(base) > max_len:
        base = base[:max_len]

    return f"{base}{ext}"

def list_images(folder: Path) -> list[Path]:
    return [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMG_EXTS]

def build_yolo_cls_dataset_from_subfolders(
    src_root_dir: str,
    out_dataset_dir: str,
    train_ratio: float = 0.8,
    seed: int = 42,
    copy: bool = True,
    min_images_per_class: int = 2,
):
    random.seed(seed)
    src_root = Path(src_root_dir)
    out_root = Path(out_dataset_dir)
    train_root = out_root / "train"
    val_root = out_root / "val"

    if not src_root.exists():
        raise FileNotFoundError(f"Source root not found: {src_root}")

    class_dirs = [d for d in src_root.iterdir() if d.is_dir()]
    if not class_dirs:
        raise FileNotFoundError(f"No class subfolders in: {src_root}")

    train_root.mkdir(parents=True, exist_ok=True)
    val_root.mkdir(parents=True, exist_ok=True)

    mover = shutil.copy2 if copy else shutil.move

    print(f"\n📁 Building dataset from: {src_root}")

    total_train = total_val = 0

    for class_dir in sorted(class_dirs, key=lambda p: p.name.lower()):
        label = normalize_label(class_dir.name)
        images = list_images(class_dir)

        if len(images) < min_images_per_class:
            print(f"[SKIP] {label} ({len(images)} images)")
            continue

        random.shuffle(images)
        n = len(images)

        val_count = max(1, int(round(n * (1 - train_ratio))))
        val_count = min(val_count, n - 1)

        train_imgs = images[:-val_count]
        val_imgs = images[-val_count:]

        (train_root / label).mkdir(exist_ok=True)
        (val_root / label).mkdir(exist_ok=True)

        for i, img in enumerate(train_imgs):
            dst = train_root / label / f"{i:05d}_{safe_filename(img.name)}"
            mover(img, dst)

        for i, img in enumerate(val_imgs):
            dst = val_root / label / f"{i:05d}_{safe_filename(img.name)}"
            mover(img, dst)

        total_train += len(train_imgs)
        total_val += len(val_imgs)

        print(f"{label}: {len(train_imgs)} train | {len(val_imgs)} val")

    print(f"✅ Output: {out_root}")
    print(f"Train: {total_train} | Val: {total_val}")


if __name__ == "__main__":

    build_yolo_cls_dataset_from_subfolders(
        src_root_dir="BODY & EXTERIOR",
        out_dataset_dir="BODY_output"
    )

    build_yolo_cls_dataset_from_subfolders(
        src_root_dir="BRAKING & SUSPENSION",
        out_dataset_dir="BRAKING_output"
    )

    build_yolo_cls_dataset_from_subfolders(
        src_root_dir="COOLING & LUBRICATION",
        out_dataset_dir="COOLING_output"
    )

    build_yolo_cls_dataset_from_subfolders(
        src_root_dir="ELECTRICAL & IGNITION",
        out_dataset_dir="ELECTRICAL_output"
    )

    build_yolo_cls_dataset_from_subfolders(
        src_root_dir="ENGINE & CORE MECHANICAL",
        out_dataset_dir="ENGINE_output"
    )

    build_yolo_cls_dataset_from_subfolders(
        src_root_dir="FUEL & AIR INTAKE",
        out_dataset_dir="FUEL_output"
    )

    build_yolo_cls_dataset_from_subfolders(
        src_root_dir="TRANSMISSION & DRIVETRAIN",
        out_dataset_dir="TRANSMISSION_output"
    )
