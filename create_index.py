"""
実験ディレクトリのファイルを読み込み、Chroma ベクターストアを構築/更新する。

- 初回実行時: 全ファイル取り込み
- 2回目以降: 前回からの差分(新規 / 変更 / 削除)のみ反映

ファイル変更検出は mtime → hash の2段階。mtime が同じなら hash は計算しない。
拡張子別の分割ルールは splitters.py を参照。
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings

import config
from splitters import split_file


# ---------- ハッシュ / マニフェスト ----------

def file_hash(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    if config.MANIFEST_PATH.exists():
        return json.loads(config.MANIFEST_PATH.read_text(encoding="utf-8"))
    return {}


def save_manifest(manifest: dict) -> None:
    config.MANIFEST_PATH.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# ---------- 対象ファイル走査 ----------

def _is_excluded(p: Path) -> bool:
    if any(part in config.EXCLUDE_DIRS for part in p.parts):
        return True
    name = p.name.lower()
    if any(name.endswith(s) for s in config.EXCLUDE_FILE_SUFFIXES):
        return True
    return False


def iter_target_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in config.TARGET_EXTENSIONS:
            continue
        if _is_excluded(p):
            continue
        try:
            if p.stat().st_size > config.MAX_FILE_BYTES:
                print(f"  ⚠ サイズ超過によりスキップ: {p}")
                continue
        except OSError:
            continue
        yield p


# ---------- 差分更新本体 ----------

def _make_chunk_ids(source: str, n: int) -> list[str]:
    """ファイルパスから決定論的にチャンク ID を発行する。"""
    base = hashlib.md5(source.encode("utf-8")).hexdigest()[:16]
    return [f"{base}::{i}" for i in range(n)]


def update_index() -> None:
    if not config.SOURCE_DIR.exists():
        raise FileNotFoundError(f"SOURCE_DIR が存在しません: {config.SOURCE_DIR}")

    manifest = load_manifest()
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )

    current = {str(p): p for p in iter_target_files(config.SOURCE_DIR)}
    prev_keys = set(manifest.keys())
    curr_keys = set(current.keys())

    new_files = curr_keys - prev_keys
    deleted_files = prev_keys - curr_keys
    common = curr_keys & prev_keys

    # 変更検出: mtime が違うものだけ hash を計算
    changed_files: set[str] = set()
    for s in common:
        path = current[s]
        try:
            mtime = path.stat().st_mtime
            if mtime == manifest[s].get("mtime"):
                continue  # 完全に未変更
            new_hash = file_hash(path)
            if new_hash != manifest[s].get("hash"):
                changed_files.add(s)
            else:
                # 内容は同じだが mtime だけずれている → manifest だけ更新
                manifest[s]["mtime"] = mtime
        except Exception as e:
            print(f"  ⚠ stat 失敗 {s}: {e}")

    print(f"[DIFF] 新規={len(new_files)}  変更={len(changed_files)}  削除={len(deleted_files)}")

    # 削除 + 変更 → Chroma から旧チャンク削除
    for s in deleted_files | changed_files:
        ids = manifest.get(s, {}).get("chunk_ids", [])
        if ids:
            try:
                vectorstore.delete(ids=ids)
            except Exception as e:
                print(f"  ⚠ 削除失敗 {s}: {e}")
        if s in deleted_files:
            manifest.pop(s, None)
            print(f"  🗑  {s}")

    # 新規 + 変更 → チャンク化して再投入
    for s in sorted(new_files | changed_files):
        path = Path(s)
        try:
            docs = split_file(path)
            if not docs:
                manifest[s] = {
                    "mtime": path.stat().st_mtime,
                    "hash": file_hash(path),
                    "chunk_ids": [],
                }
                continue
            ids = _make_chunk_ids(s, len(docs))
            vectorstore.add_documents(documents=docs, ids=ids)
            manifest[s] = {
                "mtime": path.stat().st_mtime,
                "hash": file_hash(path),
                "chunk_ids": ids,
            }
            mark = "🟢" if s in new_files else "🔄"
            print(f"  {mark} {len(docs):>3} chunks  {s}")
        except Exception as e:
            print(f"  ⚠ 取り込み失敗 {s}: {e}")

    save_manifest(manifest)
    print(f"[DONE] manifest 登録ファイル数: {len(manifest)}")


if __name__ == "__main__":
    update_index()
