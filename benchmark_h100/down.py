from pathlib import Path
import shutil
import subprocess

from huggingface_hub import snapshot_download

MODELS = [
    "Qwen/Qwen3-4B",
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
]

DOWNLOAD_ROOT = Path("/home/ubuntu/hf-model-downloads")
DRIVE_ROOT = Path("/home/ubuntu/")

CLEANUP_AFTER_ARCHIVE = True
OVERWRITE_EXISTING = False

DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
DRIVE_ROOT.mkdir(parents=True, exist_ok=True)

for repo_id in MODELS:
    model_name = repo_id.split("/")[-1]
    model_directory = DOWNLOAD_ROOT / model_name
    archive_path = DRIVE_ROOT / f"{model_name}.tar"

    if archive_path.exists() and not OVERWRITE_EXISTING:
        size_gib = archive_path.stat().st_size / (1024**3)
        print(f"Skipping existing archive: {archive_path} ({size_gib:.2f} GiB)")
        continue

    print(f"\nDownloading {repo_id}...")

    snapshot_download(
        repo_id=repo_id,
        local_dir=model_directory,
        max_workers=8,
    )

    print(f"Creating {archive_path}...")

    subprocess.run(
        [
            "tar",
            "-cf",
            str(archive_path),
            "--exclude=.cache",
            "-C",
            str(DOWNLOAD_ROOT),
            model_name,
        ],
        check=True,
    )

    print("Verifying archive...")

    subprocess.run(
        ["tar", "-tf", str(archive_path)],
        stdout=subprocess.DEVNULL,
        check=True,
    )

    size_gib = archive_path.stat().st_size / (1024**3)
    print(f"Saved: {archive_path}")
    print(f"Archive size: {size_gib:.2f} GiB")

    if CLEANUP_AFTER_ARCHIVE:
        shutil.rmtree(model_directory)
        print(f"Removed temporary directory: {model_directory}")

print("\nFinished.")
