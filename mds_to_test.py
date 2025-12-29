from pathlib import Path
from src.config import settings

DATA_PATH = settings.DATA_PATH
OUT_DIR = DATA_PATH / "txt"
OUT_DIR.mkdir(exist_ok=True)

def extract_text_from_md(path: Path) -> str:
    return path.read_text(encoding="utf-8")

def export_text_to_txt(text: str, export_path: Path) -> None:
    export_path.write_text(text, encoding="utf-8")

if __name__ == "__main__":
    md_files = list(DATA_PATH.glob("**/*.md"))
    md_files = [p for p in md_files if "txt" not in p.parts]  # undvik output-mappen

    for md_path in md_files:
        out_path = OUT_DIR / (md_path.stem + ".txt")
        export_text_to_txt(extract_text_from_md(md_path), out_path)

    print("Done.")
