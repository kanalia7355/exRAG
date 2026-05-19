# 実験ディレクトリ RAG (Ollama + LangChain v1)

ローカル完結の RAG。実験ディレクトリ配下のソースコード・メモ・論文・スライドなどを Chroma に取り込み、Gradio UI から質問できる。Zenn のローカル RAG 構築記事の構成をベースに、対応形式の拡充・ファイル差分更新・自動監視・ワンコマンド起動を追加。

## ディレクトリ構成

このプロジェクトと実験ディレクトリは**別フォルダ**で運用する。

```
projects/
├── experiment_rag/           ← この RAG プロジェクト
│   ├── config.py
│   ├── splitters.py
│   ├── create_index.py
│   ├── watch_and_update.py
│   ├── app.py
│   ├── run.py                ← ★ 普段はこれだけ実行する
│   ├── manifest.json         (自動生成)
│   └── chroma_db/            (自動生成)
│
└── experiments/              ← 既存の実験ディレクトリ(触らない)
    ├── project_a/
    └── project_b/
```

## 対応ファイル形式

| カテゴリ | 拡張子 | 取り込み戦略 | 出典メタデータ |
|---|---|---|---|
| 文章 | `.md` | 見出しで一次分割 → 日本語向け二次分割 | `h1`/`h2`/`h3` |
| 文章 | `.tex` | `\section` 等で構造的に分割 | — |
| 文章 | `.txt` | 日本語向け再帰分割 | — |
| コード | `.py` | `class`/`def`/インデントで分割 | — |
| コード | `.js` | `function`/`class`/`export` で分割 | — |
| コード | `.html` `.htm` | タグ単位で分割 | — |
| コード | `.css` | ルール境界 `}` を最優先 | — |
| 構造化 | `.json` | キー構造を保ったまま分割 | — |
| 文書 | `.pdf` | ページ単位 → 二次分割 | ページ番号 |
| 文書 | `.docx` | 全文抽出 → 日本語向け分割 | — |
| 文書 | `.pptx` | スライド単位(ノート含む) | スライド番号 |
| 文書 | `.xlsx` | シート単位 | シート名 |

## セットアップ

```bash
# 1. Ollama を入れてモデルを取得
ollama pull gemma3:4b           # 推論用
ollama pull nomic-embed-text    # 埋め込み用

# 2. Python パッケージ(コア)
pip install langchain langchain-community langchain-text-splitters \
            langchain-ollama langchain-chroma gradio watchdog

# 3. Python パッケージ(Office / PDF を扱う場合)
pip install pypdf docx2txt python-pptx openpyxl
```

外部ライブラリは `splitters.py` 内で**遅延 import** しているため、
たとえば `.xlsx` を一切使わないなら `openpyxl` を入れなくても他の形式は動く。

## 設定

`config.py` の冒頭だけ書き換える。研究室環境への切り替えもここのパスを変えるだけ。

```python
SOURCE_DIR = Path("./docs")                # ← 実験ディレクトリへのパス(絶対パス推奨)
CHROMA_DIR = Path("./chroma_db")
LLM_MODEL = "gemma3:4b"
EMBEDDING_MODEL = "nomic-embed-text"
```

絶対パス例:
```python
SOURCE_DIR = Path(r"C:\Users\Kanalia\experiments")    # Windows
SOURCE_DIR = Path("/home/kanalia/experiments")        # Linux
```

## 使い方

### 普段の起動(これだけ)

```bash
python run.py
```

これだけで次の3つが順に走る:
1. インデックス差分更新(変更されたファイルだけ取り込み直し)
2. ファイル監視をバックグラウンド起動(以降はファイル編集→自動再取り込み)
3. Gradio チャット UI 起動 → http://127.0.0.1:7860

Ctrl+C で全部停止する。

### 個別実行(必要なときだけ)

```bash
python create_index.py        # インデックス再構築/差分更新だけしたい
python watch_and_update.py    # ファイル監視だけ(UIなし)
python app.py                 # UIだけ(自動更新なし、初回起動も含まない)
```

## 仕組み

### 差分更新
`manifest.json` にファイルパス → `mtime` / `hash` / `chunk_ids` を保持。実行時に次の手順で判定する。

1. ディレクトリを走査して現在のファイル一覧を取得
2. マニフェストにあって現状に無い → **削除**(`chunk_ids` で Chroma から消す)
3. マニフェストに無い → **新規**(分割して追加)
4. 共通ファイルは `mtime` を比較 → 違えば `hash` 計算 → 違えば **変更**(旧チャンクを ID 指定で削除してから再投入)

`hash` までは内容が同じなら計算しないので、未変更ファイルが大量にあっても高速。

### ファイル監視
`watchdog` でファイルシステムイベントを購読。エディタが保存時に複数イベントを発火することがあるため、5秒のデバウンス後に `update_index()` を呼ぶ。

### サイズ・除外ルール

| 対象 | 動作 |
|---|---|
| `EXCLUDE_DIRS` に該当するディレクトリ | 走査時にスキップ |
| `EXCLUDE_FILE_SUFFIXES` に該当するファイル | 走査時にスキップ |
| `MAX_FILE_BYTES` 超のファイル | 警告ログを出してスキップ |
| 取込済みファイルが消えた | 次回更新時に Chroma からも削除 |

## 拡張

新しい拡張子に対応したい場合は3ステップ:

1. `splitters.py` に分割関数を追加
2. `splitters.py` の `SPLITTERS` 辞書に登録
3. `config.py` の `TARGET_EXTENSIONS` に追加

例: Jupyter Notebook (`.ipynb`)

```python
# splitters.py
def split_ipynb(path: Path) -> list[Document]:
    import nbformat
    nb = nbformat.read(path, as_version=4)
    docs = []
    for i, cell in enumerate(nb.cells):
        if not cell.source.strip():
            continue
        docs.append(Document(
            page_content=cell.source,
            metadata={
                "source": str(path), "filetype": "ipynb",
                "cell_type": cell.cell_type, "cell_index": i,
            },
        ))
    return docs

SPLITTERS[".ipynb"] = split_ipynb
```

## 研究室環境への移植手順

1. このフォルダ一式をコピー
2. `config.py` の `SOURCE_DIR` を研究室の実験ディレクトリへ
3. 必要なら `LLM_MODEL` を研究室マシンのスペックに合わせて変更(例: `qwen2.5:14b`, `gpt-oss:20b` など)
4. `ollama pull <モデル名>` でモデル取得
5. `python run.py` で起動

`chroma_db/` と `manifest.json` は環境ごとに自動生成されるので、コピー不要(コピーするとパスが食い違うので避ける)。
