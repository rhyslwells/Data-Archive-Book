#!/usr/bin/env python3
import shutil
import subprocess
from pathlib import Path
import argparse
import sys

# ── paths ─────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # DAB/
SCRIPTS = ROOT / "scripts"
CONTENT_DIR = ROOT / "DAB/content"
IMAGES_DIR = CONTENT_DIR / "images"

NOTES_DIR = ROOT.parent / "Notes" / "content"
# Standardised source content (absolute or relative to home)
STANDARDISED_CONTENT = Path.home() / "Desktop" / "Data-Archive" / "content" / "standardised"
STANDARDISED_IMAGES  = Path.home() / "Desktop" / "Data-Archive" / "content" / "storage" / "images"
# ── functions ─────────────────────────────────────────────────────────────

def clean_and_prepare_directories():
    if CONTENT_DIR.exists():
        shutil.rmtree(CONTENT_DIR)
    CONTENT_DIR.mkdir(parents=True)
    IMAGES_DIR.mkdir(parents=True)
    print("✅ Cleaned and recreated content directories")

def copy_standardised_content():
    NOTES_DIR.mkdir(parents=True, exist_ok=True)  # Ensure destination exists
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)  # Also good to guard this

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

def main(args):
    if args.clean:
        clean_and_prepare_directories()

    if args.copy:
        copy_standardised_content()

    if args.run:
        for script in args.run:
            run_script(script)

    if args.fallback:
        fallback = input("❓ Did an error occur in compiler.py? Run file_remover.py? (y/N): ")
        if fallback.strip().lower() == "y":
            run_script("file_remover.py")

# ── CLI entry ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build pipeline controller")
    parser.add_argument("--clean", action="store_true", help="Clean and recreate content directories")
    parser.add_argument("--copy", action="store_true", help="Copy standardised content and images")
    parser.add_argument("--run", nargs="*", help="Run script(s) in scripts directory (e.g. selected_files.py)")
    parser.add_argument("--fallback", action="store_true", help="Prompt to run fallback script")

    if len(sys.argv) == 1:
        parser.print_help()
    else:
        args = parser.parse_args()
        main(args)

# python3 main.py --clean
# python3 main.py --copy
# python3 main.py --run selected_files.py
# python3 main.py --run update.py
# python3 main.py --run compiler.py
# python3 main.py --fallback