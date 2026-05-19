"""
ワンコマンド起動スクリプト。次の3ステップをまとめて実行する。

  1. インデックスの差分更新 (create_index.update_index)
  2. ファイル監視のバックグラウンド起動 (watch_and_update.IndexUpdater)
  3. Gradio チャット UI の起動 (app.launch)

実験ディレクトリで作業中もこの1プロセスで全部回り続けるので、
普段の起動は `python run.py` だけで良い。
Ctrl+C で全停止する。
"""
from __future__ import annotations

import threading
import time

from watchdog.observers import Observer

import config
import app
from create_index import update_index
from watch_and_update import IndexUpdater


def _start_watcher_background() -> Observer:
    """ファイル監視を別スレッドで動かす。"""
    handler = IndexUpdater()
    observer = Observer()
    observer.schedule(handler, str(config.SOURCE_DIR), recursive=True)
    observer.start()

    def loop():
        while True:
            handler.tick()
            time.sleep(1)

    threading.Thread(target=loop, daemon=True).start()
    print(f"[WATCH] バックグラウンド監視開始: {config.SOURCE_DIR}")
    return observer


def main() -> None:
    line = "=" * 60
    print(line)
    print(" 実験ディレクトリ RAG 起動")
    print(line)

    print("\n[1/3] インデックス差分更新")
    update_index()

    print("\n[2/3] ファイル監視をバックグラウンドで起動")
    _start_watcher_background()

    print("\n[3/3] チャット UI を起動 (Ctrl+C で停止)\n")
    app.launch()


if __name__ == "__main__":
    main()
