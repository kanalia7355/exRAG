"""
プロジェクト全体の設定。研究室環境に切り替えるときは
このファイルの冒頭(SOURCE_DIR など)を書き換えるだけで良い。
"""
from pathlib import Path

# ====== ここを書き換えれば研究室環境にも対応できる ======

# RAG の参照対象ディレクトリ(実験ディレクトリ)
SOURCE_DIR = Path(r"C:\Users\hirot\lab_ex")

# ベクターストアの永続化ディレクトリ
CHROMA_DIR = Path("./chroma_db")

# 差分更新用マニフェスト(ファイルのハッシュとチャンク ID を保持)
MANIFEST_PATH = Path("./manifest.json")

# Ollama モデル
# 軽量だが回答品質が低い: "gemma3:4b"
# RTX 4070 + 32GB RAM 推奨: "qwen2.5:14b" / "qwen2.5-coder:14b"
# それでも重い場合: "qwen2.5:7b"
LLM_MODEL = "qwen2.5:14b"
EMBEDDING_MODEL = "nomic-embed-text"

# LLM 推論パラメータ
LLM_TEMPERATURE = 0.2      # 低いほど一貫した回答に。0 〜 0.4 推奨
LLM_NUM_CTX = 8192         # コンテキスト長。大きいモデル + 多検索結果なら 12288 など

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
MAX_FILE_BYTES = 20 * 1024 * 1024   # 20 MB

# ---------- 検索パラメータ ----------

# 類似検索で取得する件数。多いほど回答に厚みが出るが、
# コンテキストウィンドウを圧迫する。8 〜 12 が目安。
TOP_K = 8

# 検索方法
#   "similarity" : 純粋なベクトル類似度
#   "mmr"        : 多様性を考慮(複数の出典が出やすく、引用の網羅性が上がる)
SEARCH_TYPE = "mmr"

# MMR のときに内部で候補として取り出す件数(TOP_K の2〜4倍が目安)
MMR_FETCH_K = 24