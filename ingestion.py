from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional, Dict, Any, List

import lancedb
from google import genai

from backend.constants import VECTOR_DATABASE_PATH, DATA_PATH
from backend.config import settings
from backend.data_models import Article, EMBEDDING_DIM

TABLE_NAME = "articles"
MANIFEST_NAME = ".ingest_manifest.json"


def connect_vector_db(db_path: str | Path) -> lancedb.DBConnection:
    db_path = Path(db_path)
    db_path.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(uri=str(db_path))
    db.create_table(TABLE_NAME, schema=Article, exist_ok=True)
    return db


def _manifest_path(data_dir: Path) -> Path:
    return data_dir / MANIFEST_NAME


def _load_manifest(data_dir: Path) -> Dict[str, Dict[str, Any]]:
    p = _manifest_path(data_dir)
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_manifest(data_dir: Path, manifest: Dict[str, Dict[str, Any]]) -> None:
    p = _manifest_path(data_dir)
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")


def _safe_sql_string(value: str) -> str:
    return value.replace("'", "''")


def embed_text(client: genai.Client, text: str) -> List[float]:
    res = client.models.embed_content(
        model=settings.embed_model,   # "models/text-embedding-004"
        contents=text,
    )
    vec = res.embeddings[0].values
    if len(vec) != EMBEDDING_DIM:
        raise ValueError(
            f"ابعاد embedding ناهماهنگ است: {len(vec)} به‌جای {EMBEDDING_DIM}. "
            f"settings.embed_model و EMBEDDING_DIM را بررسی کن."
        )
    return vec


def upsert_txt_files(
    table,
    data_dir: Path,
    *,
    glob_pattern: str = "*.txt",
    encoding: str = "utf-8",
    sleep_seconds: float = 0.0,
    verbose: bool = True,
) -> int:
    ingested = 0

    if not data_dir.exists():
        raise FileNotFoundError(f"DATA_PATH وجود ندارد: {data_dir}")

    client = genai.Client(api_key=settings.api_key)
    manifest = _load_manifest(data_dir)

    for file_path in sorted(data_dir.glob(glob_pattern)):
        if not file_path.is_file():
            continue
        if file_path.name == MANIFEST_NAME:
            continue

        doc_id = file_path.stem
        stat = file_path.stat()
        mtime_ns = stat.st_mtime_ns
        size = stat.st_size

        prev = manifest.get(doc_id)
        unchanged = prev is not None and prev.get("mtime_ns") == mtime_ns and prev.get("size") == size
        if unchanged:
            if verbose:
                print(f"رد شد (بدون تغییر): {file_path.name}")
            continue

        try:
            content = file_path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            content = file_path.read_text(encoding=encoding, errors="replace")

        embedding = embed_text(client, content)

        safe_doc_id = _safe_sql_string(doc_id)
        table.delete(f"doc_id = '{safe_doc_id}'")

        table.add(
            [
                {
                    "doc_id": doc_id,
                    "filepath": str(file_path),
                    "filename": file_path.stem,
                    "content": content,
                    "embedding": embedding,
                }
            ]
        )

        manifest[doc_id] = {"mtime_ns": mtime_ns, "size": size, "filename": file_path.name}
        _save_manifest(data_dir, manifest)

        ingested += 1
        if verbose:
            print(f"Ingested: {file_path.name} (doc_id={doc_id})")

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    if verbose:
        print(f"پایان. فایل‌های جدید/به‌روزشده: {ingested}")
        print(f"Manifest: {_manifest_path(data_dir)}")

    return ingested


def main(mode: Optional[str] = None) -> None:
    db = connect_vector_db(VECTOR_DATABASE_PATH)
    table = db[TABLE_NAME]
    mode = (mode or "once").lower()

    if mode == "watch":
        while True:
            try:
                count = upsert_txt_files(table, DATA_PATH, verbose=True)
                print(f"چرخه تمام شد. {count} فایل به‌روزرسانی شد. خواب ۳۰ ثانیه…")
            except Exception as e:
                print(f"[watch] خطا: {e}")
            time.sleep(30.0)
    else:
        upsert_txt_files(table, DATA_PATH, verbose=True)


if __name__ == "__main__":
    main("once")
