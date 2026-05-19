"""
プロジェクト全体の設定。研究室環境に切り替えるときは
このファイルの冒頭(SOURCE_DIR など)を書き換えるだけで良い。
"""
from pathlib import Path

# ====== ここを書き換えれば研究室環境にも対応できる ======

# RAG の参照対象ディレクトリ(実験ディレクトリ)
SOURCE_DIR = Path("./docs")

# ベクターストアの永続化ディレクトリ
CHROMA_DIR = Path("./chroma_db")

# 差分更新用マニフェスト(ファイルのハッシュとチャンク ID を保持)
MANIFEST_PATH = Path("./manifest.json")

# Ollama モデル
LLM_MODEL = "gemma3:4b"                  # 推論用。gpt-oss など他モデルへ差替え可
EMBEDDING_MODEL = "nomic-embed-text"     # 埋め込み用

# ========================================================

# 取り込み対象の拡張子
# (新しい形式を増やす場合は splitters.py の SPLITTERS にも登録する)
TARGET_EXTENSIONS = {
    # 文章
    ".md", ".txt", ".tex",
    # コード
    ".py", ".js", ".html", ".htm", ".css",
    # 構造化
    ".json",
    # Office / PDF
    ".pdf", ".docx", ".pptx", ".xlsx",
}

# 走査時に無視するディレクトリ名
EXCLUDE_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "env",
    "node_modules", ".ipynb_checkpoints",
    "dist", "build", ".next", ".cache",
    "chroma_db",        # 自分自身の永続化先は除外
}

# 走査時に無視するファイル名(末尾一致)
EXCLUDE_FILE_SUFFIXES = (
    ".min.js", ".min.css",     # ミニファイ済みは取り込まない
    "package-lock.json",
    "yarn.lock",
)

# チャンク分割パラメータ(文章系)
CHUNK_SIZE_TEXT = 800
CHUNK_OVERLAP_TEXT = 120

# チャンク分割パラメータ(コード系)
CHUNK_SIZE_CODE = 1200
CHUNK_OVERLAP_CODE = 150

# 単一ファイルの最大サイズ(これより大きいファイルはスキップ)
# PDF や XLSX は大きくなりやすいので少し余裕を持たせる
MAX_FILE_BYTES = 20 * 1024 * 1024   # 20 MB

# 類似検索で取得する件数
TOP_K = 6
