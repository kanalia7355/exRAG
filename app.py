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


def build_agent():
    """エージェント本体を構築して返す。"""
    model = ChatOllama(model=config.LLM_MODEL)
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    vector_store = Chroma(
        embedding_function=embeddings,
        persist_directory=str(config.CHROMA_DIR),
    )

    @dynamic_prompt
    def prompt_with_context(request: ModelRequest) -> str:
        last_query = request.state["messages"][-1].text
        retrieved = vector_store.similarity_search(last_query, k=config.TOP_K)

        docs_content = "\n\n---\n\n".join(
            f"[出典: {d.metadata.get('source', 'unknown')}"
            f" / {d.metadata.get('filetype', '')}]\n{d.page_content}"
            for d in retrieved
        )

        return (
            "あなたはユーザの実験ディレクトリの内容(コードとメモ)に詳しいアシスタントです。"
            "以下のコンテキストのみを根拠に、日本語で簡潔かつ正確に回答してください。"
            "コードを引用するときは出典ファイルパスを明示してください。"
            "コンテキストに含まれていない事項は推測せず、"
            "『資料に記載なし』と述べてください。\n\n"
            f"# コンテキスト\n{docs_content}"
        )

    return create_agent(model, tools=[], middleware=[prompt_with_context])


def launch() -> None:
    """Gradio UI を起動する。"""
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
            f"Embedding: `{config.EMBEDDING_MODEL}`"
        ),
    )
    demo.launch()


if __name__ == "__main__":
    launch()
