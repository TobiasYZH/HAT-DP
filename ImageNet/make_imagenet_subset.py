import argparse
import csv
import io
import json
import os
import random
import shutil
import tarfile
import zipfile
from pathlib import Path, PurePosixPath


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create a class-balanced ImageNet subset in torchvision ImageFolder format. "
            "The expected output layout is: out_root/train/<class_id>/*.JPEG. "
            "The source can be either an extracted ImageNet train folder or the official "
            "ILSVRC2012_img_train.tar containing per-class tar files."
        )
    )
    parser.add_argument(
        "--src-train",
        type=str,
        default="ILSVRC2012_img_train.tar",
        help=(
            "Path to the extracted ImageNet train folder with 1000 class subfolders, "
            "or path to the official ILSVRC2012_img_train.tar."
        ),
    )
    parser.add_argument(
        "--out-root",
        type=str,
        default=None,
        help="Output root. The script will create out_root/train, manifest.csv, and metadata.json.",
    )
    parser.add_argument("--images-per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--mode",
        type=str,
        default="copy",
        choices=["copy", "hardlink", "symlink"],
        help=(
            "How to place files when --src-train is an extracted folder. "
            "Tar input always writes extracted image bytes."
        ),
    )
    parser.add_argument(
        "--archive-path",
        type=str,
        default=None,
        help="Optional archive path, e.g. imagenet_subset_100.tar or imagenet_subset_300.zip.",
    )
    parser.add_argument(
        "--archive-only",
        action="store_true",
        help=(
            "Write the subset directly into --archive-path without creating an output folder. "
            "This mode is intended for the official ILSVRC2012_img_train.tar input."
        ),
    )
    parser.add_argument(
        "--archive-root-name",
        type=str,
        default=None,
        help=(
            "Top-level folder name stored inside --archive-path when using --archive-only. "
            "Defaults to the archive filename without suffixes."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the existing output root before creating the subset.",
    )
    parser.add_argument("--expected-classes", type=int, default=1000)
    parser.add_argument(
        "--allow-class-mismatch",
        action="store_true",
        help="Continue even if the number of classes is not expected-classes.",
    )
    parser.add_argument(
        "--allow-fewer-images",
        action="store_true",
        help="Copy/extract all available images if a class has fewer than images-per-class.",
    )
    return parser.parse_args()


def is_image_name(name):
    return PurePosixPath(str(name)).suffix.lower() in IMAGE_EXTENSIONS


def list_images(class_dir):
    return sorted(
        path for path in class_dir.iterdir()
        if path.is_file() and is_image_name(path.name)
    )


def prepare_output(out_root, overwrite):
    if out_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {out_root}. "
                "Use --overwrite or choose a new --out-root."
            )
        shutil.rmtree(out_root)
    (out_root / "train").mkdir(parents=True, exist_ok=True)


def place_file(src, dst, mode):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(src, dst)
    elif mode == "hardlink":
        os.link(src, dst)
    elif mode == "symlink":
        os.symlink(src, dst)
    else:
        raise ValueError(f"Unsupported mode: {mode}")


def make_archive(out_root, archive_path):
    archive_path = Path(archive_path).expanduser().resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)

    root_name = out_root.name
    suffixes = "".join(archive_path.suffixes).lower()

    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_STORED) as zf:
            for path in sorted(out_root.rglob("*")):
                zf.write(path, arcname=Path(root_name) / path.relative_to(out_root))
        return

    if suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
        mode = "w:gz"
    elif suffixes.endswith(".tar.bz2"):
        mode = "w:bz2"
    elif suffixes.endswith(".tar.xz"):
        mode = "w:xz"
    elif suffixes.endswith(".tar"):
        mode = "w"
    else:
        raise ValueError(
            "Unsupported archive extension. Use .tar, .tar.gz, .tgz, .tar.bz2, .tar.xz, or .zip."
        )

    with tarfile.open(archive_path, mode) as tf:
        tf.add(out_root, arcname=root_name)


def get_tar_write_mode(archive_path):
    suffixes = "".join(Path(archive_path).suffixes).lower()
    if suffixes.endswith(".tar.gz") or suffixes.endswith(".tgz"):
        return "w:gz"
    if suffixes.endswith(".tar.bz2"):
        return "w:bz2"
    if suffixes.endswith(".tar.xz"):
        return "w:xz"
    if suffixes.endswith(".tar"):
        return "w"
    raise ValueError("Archive-only mode supports .tar, .tar.gz, .tgz, .tar.bz2, and .tar.xz.")


def archive_root_name(archive_path, explicit_root_name):
    if explicit_root_name:
        return explicit_root_name.strip("/\\")

    path = Path(archive_path)
    name = path.name
    for suffix in path.suffixes:
        if name.endswith(suffix):
            name = name[: -len(suffix)]
    return name or "imagenet_subset"


def validate_class_count(num_classes, expected_classes, allow_class_mismatch, source):
    if num_classes != expected_classes and not allow_class_mismatch:
        raise RuntimeError(
            f"Expected {expected_classes} classes, found {num_classes} in {source}. "
            "If this is intentional, pass --allow-class-mismatch."
        )


def select_from_extracted_dir(class_dir, images_per_class, allow_fewer_images, rng):
    images = list_images(class_dir)
    if len(images) < images_per_class:
        if not allow_fewer_images:
            raise RuntimeError(
                f"Class {class_dir.name} only has {len(images)} images, "
                f"but --images-per-class is {images_per_class}."
            )
        selected = images
    else:
        selected = sorted(rng.sample(images, images_per_class))
    return selected, len(images)


def create_subset_from_extracted_dir(args, src_train, out_root, rng):
    class_dirs = sorted(path for path in src_train.iterdir() if path.is_dir())
    tar_files = sorted(src_train.glob("*.tar"))
    if not class_dirs and tar_files:
        raise RuntimeError(
            "No class folders were found, but .tar files were detected. "
            "Pass the official ILSVRC2012_img_train.tar itself to --src-train, "
            "or extract these class tar files into class folders first."
        )

    validate_class_count(
        len(class_dirs),
        args.expected_classes,
        args.allow_class_mismatch,
        src_train,
    )

    manifest_rows = []
    total_images = 0

    for class_dir in class_dirs:
        selected, available = select_from_extracted_dir(
            class_dir,
            args.images_per_class,
            args.allow_fewer_images,
            rng,
        )
        dst_class_dir = out_root / "train" / class_dir.name
        for idx, src_path in enumerate(selected):
            dst_path = dst_class_dir / src_path.name
            place_file(src_path, dst_path, args.mode)
            manifest_rows.append(
                {
                    "class_id": class_dir.name,
                    "index_in_class": idx,
                    "source_path": str(src_path),
                    "relative_output_path": str(dst_path.relative_to(out_root)),
                }
            )
        total_images += len(selected)
        print(f"{class_dir.name}: selected {len(selected)} / {available}")

    return manifest_rows, total_images, len(class_dirs), "extracted_dir"


def class_id_from_tar_member(member_name):
    name = PurePosixPath(member_name).name
    if not name.endswith(".tar"):
        raise ValueError(f"Expected a class tar member, got: {member_name}")
    return name[:-4]


def list_class_tar_members(train_tar):
    with tarfile.open(train_tar, "r:*") as outer:
        return sorted(
            (
                member
                for member in outer.getmembers()
                if member.isfile() and PurePosixPath(member.name).name.endswith(".tar")
            ),
            key=lambda member: class_id_from_tar_member(member.name),
        )


def safe_image_basename(member_name):
    basename = PurePosixPath(member_name).name
    if not basename or basename in {".", ".."}:
        raise RuntimeError(f"Unsafe image member name in tar: {member_name}")
    return basename


def reservoir_sample_tar_images(class_fileobj, images_per_class, rng):
    selected = []
    seen = 0

    with tarfile.open(fileobj=class_fileobj, mode="r|*") as inner:
        for member in inner:
            if not member.isfile() or not is_image_name(member.name):
                continue

            seen += 1
            keep_index = None
            if len(selected) < images_per_class:
                keep_index = len(selected)
            else:
                candidate_index = rng.randrange(seen)
                if candidate_index < images_per_class:
                    keep_index = candidate_index

            if keep_index is None:
                continue

            extracted = inner.extractfile(member)
            if extracted is None:
                raise RuntimeError(f"Failed to read image member from class tar: {member.name}")

            item = {
                "name": safe_image_basename(member.name),
                "source_member": member.name,
                "data": extracted.read(),
            }
            if keep_index == len(selected):
                selected.append(item)
            else:
                selected[keep_index] = item

    return sorted(selected, key=lambda item: item["name"]), seen


def unique_output_path(dst_class_dir, filename, used_names):
    candidate = filename
    stem = PurePosixPath(filename).stem
    suffix = PurePosixPath(filename).suffix
    counter = 1
    while candidate in used_names:
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    used_names.add(candidate)
    return dst_class_dir / candidate


def create_subset_from_train_tar(args, src_train, out_root, rng):
    if args.mode != "copy":
        raise ValueError("--mode only applies to extracted-folder input. Tar input writes extracted copies.")

    class_members = list_class_tar_members(src_train)
    validate_class_count(
        len(class_members),
        args.expected_classes,
        args.allow_class_mismatch,
        src_train,
    )

    manifest_rows = []
    total_images = 0

    with tarfile.open(src_train, "r:*") as outer:
        for class_member in class_members:
            class_id = class_id_from_tar_member(class_member.name)
            class_fileobj = outer.extractfile(class_member)
            if class_fileobj is None:
                raise RuntimeError(f"Failed to read class tar from train tar: {class_member.name}")

            with class_fileobj:
                selected, available = reservoir_sample_tar_images(
                    class_fileobj,
                    args.images_per_class,
                    rng,
                )

            if available < args.images_per_class and not args.allow_fewer_images:
                raise RuntimeError(
                    f"Class {class_id} only has {available} images, "
                    f"but --images-per-class is {args.images_per_class}."
                )
            if available < args.images_per_class:
                selected = selected[:available]

            dst_class_dir = out_root / "train" / class_id
            dst_class_dir.mkdir(parents=True, exist_ok=True)
            used_names = set()

            for idx, item in enumerate(selected):
                dst_path = unique_output_path(dst_class_dir, item["name"], used_names)
                dst_path.write_bytes(item["data"])
                manifest_rows.append(
                    {
                        "class_id": class_id,
                        "index_in_class": idx,
                        "source_path": f"{src_train}::{class_member.name}::{item['source_member']}",
                        "relative_output_path": str(dst_path.relative_to(out_root)),
                    }
                )

            total_images += len(selected)
            print(f"{class_id}: selected {len(selected)} / {available}")

    return manifest_rows, total_images, len(class_members), "official_train_tar"


def add_bytes_to_tar(tf, arcname, data):
    info = tarfile.TarInfo(str(PurePosixPath(arcname)))
    info.size = len(data)
    info.mode = 0o644
    tf.addfile(info, io.BytesIO(data))


def add_manifest_to_tar(tf, root_name, manifest_rows):
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=["class_id", "index_in_class", "source_path", "relative_output_path"],
    )
    writer.writeheader()
    writer.writerows(manifest_rows)
    add_bytes_to_tar(
        tf,
        PurePosixPath(root_name) / "manifest.csv",
        buffer.getvalue().encode("utf-8"),
    )


def add_metadata_to_tar(tf, root_name, metadata):
    add_bytes_to_tar(
        tf,
        PurePosixPath(root_name) / "metadata.json",
        json.dumps(metadata, indent=2).encode("utf-8"),
    )


def create_archive_only_from_train_tar(args, src_train):
    if not args.archive_path:
        raise ValueError("--archive-only requires --archive-path.")
    if not src_train.is_file() or src_train.suffix.lower() != ".tar":
        raise RuntimeError("--archive-only currently expects the official ILSVRC2012_img_train.tar input.")
    if args.mode != "copy":
        raise ValueError("--mode only applies when creating an output folder.")

    archive_path = Path(args.archive_path).expanduser().resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    root_name = archive_root_name(archive_path, args.archive_root_name)
    rng = random.Random(args.seed)

    class_members = list_class_tar_members(src_train)
    validate_class_count(
        len(class_members),
        args.expected_classes,
        args.allow_class_mismatch,
        src_train,
    )

    manifest_rows = []
    total_images = 0

    with tarfile.open(src_train, "r:*") as outer, tarfile.open(
        archive_path,
        get_tar_write_mode(archive_path),
    ) as out_tf:
        for class_member in class_members:
            class_id = class_id_from_tar_member(class_member.name)
            class_fileobj = outer.extractfile(class_member)
            if class_fileobj is None:
                raise RuntimeError(f"Failed to read class tar from train tar: {class_member.name}")

            with class_fileobj:
                selected, available = reservoir_sample_tar_images(
                    class_fileobj,
                    args.images_per_class,
                    rng,
                )

            if available < args.images_per_class and not args.allow_fewer_images:
                raise RuntimeError(
                    f"Class {class_id} only has {available} images, "
                    f"but --images-per-class is {args.images_per_class}."
                )

            used_names = set()
            for idx, item in enumerate(selected):
                output_name = unique_archive_filename(item["name"], used_names)
                relative_output_path = str(PurePosixPath("train") / class_id / output_name)
                add_bytes_to_tar(
                    out_tf,
                    PurePosixPath(root_name) / relative_output_path,
                    item["data"],
                )
                manifest_rows.append(
                    {
                        "class_id": class_id,
                        "index_in_class": idx,
                        "source_path": f"{src_train}::{class_member.name}::{item['source_member']}",
                        "relative_output_path": relative_output_path,
                    }
                )

            total_images += len(selected)
            print(f"{class_id}: selected {len(selected)} / {available}")

        metadata = {
            "source_train": str(src_train),
            "source_format": "official_train_tar",
            "archive_path": str(archive_path),
            "archive_root_name": root_name,
            "images_per_class": args.images_per_class,
            "seed": args.seed,
            "mode": "archive_only",
            "num_classes": len(class_members),
            "total_images": total_images,
            "layout": f"ImageFolder: {root_name}/train/<class_id>/*",
        }
        add_manifest_to_tar(out_tf, root_name, manifest_rows)
        add_metadata_to_tar(out_tf, root_name, metadata)

    print("Done.")
    print(f"Archive     : {archive_path}")
    print(f"Archive root: {root_name}")
    print(f"Train folder: {root_name}/train")
    print(f"Images      : {total_images}")


def unique_archive_filename(filename, used_names):
    candidate = filename
    stem = PurePosixPath(filename).stem
    suffix = PurePosixPath(filename).suffix
    counter = 1
    while candidate in used_names:
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    used_names.add(candidate)
    return candidate


def write_manifest_and_metadata(args, src_train, out_root, manifest_rows, total_images, num_classes, source_format):
    manifest_path = out_root / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_id", "index_in_class", "source_path", "relative_output_path"],
        )
        writer.writeheader()
        writer.writerows(manifest_rows)

    metadata = {
        "source_train": str(src_train),
        "source_format": source_format,
        "output_root": str(out_root),
        "images_per_class": args.images_per_class,
        "seed": args.seed,
        "mode": args.mode,
        "num_classes": num_classes,
        "total_images": total_images,
        "layout": "ImageFolder: out_root/train/<class_id>/*",
    }
    metadata_path = out_root / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return manifest_path, metadata_path


def main():
    args = parse_args()
    src_train = Path(args.src_train).expanduser().resolve()

    if not src_train.exists():
        raise FileNotFoundError(f"Source train path does not exist: {src_train}")

    if args.archive_only:
        create_archive_only_from_train_tar(args, src_train)
        return

    if not args.out_root:
        raise ValueError("--out-root is required unless --archive-only is used.")

    out_root = Path(args.out_root).expanduser().resolve()
    prepare_output(out_root, args.overwrite)
    rng = random.Random(args.seed)

    if src_train.is_dir():
        manifest_rows, total_images, num_classes, source_format = create_subset_from_extracted_dir(
            args,
            src_train,
            out_root,
            rng,
        )
    elif src_train.is_file() and src_train.suffix.lower() == ".tar":
        manifest_rows, total_images, num_classes, source_format = create_subset_from_train_tar(
            args,
            src_train,
            out_root,
            rng,
        )
    else:
        raise RuntimeError(
            "--src-train must be an extracted ImageNet train directory or the official "
            "ILSVRC2012_img_train.tar file."
        )

    manifest_path, metadata_path = write_manifest_and_metadata(
        args,
        src_train,
        out_root,
        manifest_rows,
        total_images,
        num_classes,
        source_format,
    )

    if args.archive_path:
        print(f"Creating archive: {args.archive_path}")
        make_archive(out_root, args.archive_path)

    print("Done.")
    print(f"Subset root : {out_root}")
    print(f"Train folder: {out_root / 'train'}")
    print(f"Images      : {total_images}")
    print(f"Manifest    : {manifest_path}")
    print(f"Metadata    : {metadata_path}")
    if args.archive_path:
        print(f"Archive     : {Path(args.archive_path).expanduser().resolve()}")


if __name__ == "__main__":
    main()
