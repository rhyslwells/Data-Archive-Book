#!/usr/bin/env python3
import shutil
import subprocess
from pathlib import Path

# ── paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # DAB/
SCRIPTS = ROOT / "scripts"
CONTENT_DIR = ROOT / "content"
IMAGES_DIR = CONTENT_DIR / "images"

NOTES_DIR = ROOT.parent / "Notes" / "content"
STANDARDISED_CONTENT = Path(r".\Desktop\Data-Archive\content\standardised")
STANDARDISED_IMAGES  = Path(r".\Desktop\Data-Archive\content\storage\images")

# ── functions ─────────────────────────────────────────────────────────────
def clean_and_prepare_directories():
    if CONTENT_DIR.exists():
        shutil.rmtree(CONTENT_DIR)
    CONTENT_DIR.mkdir(parents=True)
    IMAGES_DIR.mkdir(parents=True)
    print("✅ Cleaned and recreated content directories")

def copy_standardised_content():
    for file in STANDARDISED_CONTENT.glob("*.md"):
        shutil.copy(file, NOTES_DIR)
    for image in STANDARDISED_IMAGES.glob("*"):
        shutil.copy(image, IMAGES_DIR)
    print("📄 Copied standardised markdown files")
    print("🖼️  Copied images to DAB/content/images")

def run_script(script_name):
    script_path = SCRIPTS / script_name
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    if result.returncode == 0:
        print(f"✅ {script_name} completed")
    else:
        print(f"❌ Error in {script_name}:\n{result.stdout}\n{result.stderr}")

# ── execution ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    clean_and_prepare_directories()
    copy_standardised_content()
    run_script("selected_files.py")
    run_script("update.py")
    run_script("compiler.py")

    # optionally run fallback if compiler failed (manually or with logic)
    fallback = input("❓ Did an error occur in compiler.py? Run file_remover.py? (y/N): ")
    if fallback.strip().lower() == "y":
        run_script("file_remover.py")
