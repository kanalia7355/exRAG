"""
Gradio 製のチャット UI。
LangChain v1 の create_agent + @dynamic_prompt ミドルウェアで
検索文書をシステムプロンプトに注入する形の RAG。

`python app.py` で UI 単体起動可、
`run.py` から `app.launch()` 呼び出しでも起動可。
"""
import gradio as gr
from langchain.agents import create_agent
from langchain.agents.middleware import ModelRequest, dynamic_prompt
from langchain_chroma import Chroma
from langchain_ollama import ChatOllama, OllamaEmbeddings

import config


# ---------- 出典フォーマッタ ----------

def _format_citation(meta: dict) -> str:
    """metadata からコンパクトな出典文字列を組み立てる。
    例: 'docs/notes.md | md | §実験概要'
        'docs/paper.pdf | pdf | p.3'
        'docs/deck.pptx | pptx | slide 5'
        'docs/data.xlsx | xlsx | sheet 結果'
    """
    parts = [str(meta.get("source", "unknown"))]
    ftype = meta.get("filetype")
    if ftype:
        parts.append(ftype)

    if "page" in meta:
        parts.append(f"p.{meta['page']}")
    if "slide" in meta:
        parts.append(f"slide {meta['slide']}")
    if "sheet" in meta:
        parts.append(f"sheet {meta['sheet']}")
    # Markdown 見出し
    for h in ("h3", "h2", "h1"):
        if h in meta:
            parts.append(f"§{meta[h]}")
            break

    return " | ".join(parts)


# ---------- システムプロンプト ----------

SYSTEM_PROMPT_TEMPLATE = """\
あなたはユーザの実験ディレクトリ(コード・メモ・論文・スライド等)の内容に精通した
リサーチアシスタントです。以下のコンテキストを根拠に、丁寧かつ具体的に回答してください。

# 回答の作法(厳守)

1. **十分に詳しく答える**: 関連する事実・コード・式・図表の言及を省略しない。
   2行で済ませず、必要な背景・前提・派生情報まで踏み込んで説明する。
2. **構造化して書く**: 必要に応じて見出し / 箇条書き / コードブロックを使い、読みやすくする。
   コードは ```python のように言語付きフェンスで囲む。
3. **すべての主要な主張に [S番号] の出典タグを付ける**:
   - 例: 「この関数は MiDaS で深度推定する [S2]。チャネル順序は BGR → RGB に変換される [S3]。」
   - 1文に複数の出典がある場合は [S2][S5] のように並べる。
4. **回答末尾に必ず `## 参照` セクションを置く**: 使った Source を箇条書きで列挙する。
   - 形式: `- [S2] docs/foo.py | py`
5. **資料に無いことは推測しない**: 「資料に記載がありません」と明示する。
   それでも一般論で補足できる場合は『資料外の一般論』と前置きしてから書く。
6. **質問が曖昧なときは複数の解釈を提示する**: それぞれに対して短く答え、
   どの方向で深掘りすればよいか確認する。

# コンテキスト

{docs_content}
"""

NO_HIT_PROMPT = """\
あなたはユーザの実験ディレクトリに詳しいアシスタントです。
ただし、現在のインデックスから関連する情報を見つけられませんでした。

ユーザに以下を伝えてください:
- 検索結果が0件であったこと
- 質問の言い換えや具体化(ファイル名・関数名・キーワードの明示)を提案
- 該当ファイルがまだインデックスに含まれていない可能性があれば、
  `python create_index.py` で再構築できることを案内
"""


# ---------- エージェント構築 ----------

def _retrieve(vector_store: Chroma, query: str):
    if config.SEARCH_TYPE == "mmr":
        return vector_store.max_marginal_relevance_search(
            query, k=config.TOP_K, fetch_k=config.MMR_FETCH_K,
        )
    return vector_store.similarity_search(query, k=config.TOP_K)


def build_agent():
    """エージェント本体を構築して返す。"""
    model = ChatOllama(
        model=config.LLM_MODEL,
        temperature=config.LLM_TEMPERATURE,
        num_ctx=config.LLM_NUM_CTX,
    )
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )

    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:
        last_query = request.state["messages"][-1].text
        retrieved = _retrieve(vector_store, last_query)

        if not retrieved:
            return NO_HIT_PROMPT

        blocks = []
        for i, d in enumerate(retrieved, start=1):
            citation = _format_citation(d.metadata)
            blocks.append(f"## [S{i}] {citation}\n\n{d.page_content}")
        docs_content = "\n\n---\n\n".join(blocks)

        return SYSTEM_PROMPT_TEMPLATE.format(docs_content=docs_content)

    return create_agent(model, tools=[], middleware=[prompt_with_context])


# ---------- Gradio 起動 ----------

def launch() -> None:
    agent = build_agent()

    def predict(message: str, history):
        partial = ""
        for token, _meta in agent.stream(
            {"messages": [{"role": "user", "content": message}]},
            stream_mode="messages",
        ):
            if token.content_blocks:
                partial += token.content_blocks[0]["text"]
                yield partial

    demo = gr.ChatInterface(
        predict,
        title="実験ディレクトリ RAG",
        description=(
            f"参照先: `{config.SOURCE_DIR}` / "
            f"LLM: `{config.LLM_MODEL}` / "
            f"検索: `{config.SEARCH_TYPE}` (k={config.TOP_K})"
        ),
    )
    demo.launch()


if __name__ == "__main__":
    launch()