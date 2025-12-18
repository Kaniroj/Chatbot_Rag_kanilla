from pathlib import Path
from backend.constants import DATA_PATH


def extract_text_from_md(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def export_text_to_txt(text: str, export_path: Path) -> None:
    export_path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    md_files = list(DATA_PATH.glob("**/*.md"))

    print(f"DATA_PATH = {DATA_PATH}")
    print(f"Found {len(md_files)} .md files")

    for md_path in md_files:
        out_path = md_path.with_suffix(".txt")  # same folder, same name, .txt
        print("Converting:", md_path.relative_to(DATA_PATH))
        export_text_to_txt(extract_text_from_md(md_path), out_path)

    print("Done.")
