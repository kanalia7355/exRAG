"""
SOURCE_DIR を監視し、対象ファイルに変更があったら自動で update_index() を実行する。

エディタの保存タイミングで複数イベントが連続発火することがあるため、
最後のイベントから DEBOUNCE_SECONDS だけ静かになってから更新を走らせる。
"""
from __future__ import annotations

import time
from pathlib import Path

from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

import config
from create_index import update_index

# 連続イベントをまとめるための待機秒
DEBOUNCE_SECONDS = 5.0


class IndexUpdater(FileSystemEventHandler):
    def __init__(self) -> None:
        self._last_event_at = 0.0
        self._dirty = False

    def _is_target(self, path_str: str) -> bool:
        p = Path(path_str)
        if p.suffix.lower() not in config.TARGET_EXTENSIONS:
            return False
        if any(part in config.EXCLUDE_DIRS for part in p.parts):
            return False
        return True

    def on_any_event(self, event):
        if event.is_directory:
            return
        if not self._is_target(event.src_path):
            return
        self._dirty = True
        self._last_event_at = time.time()
        print(f"  [event] {event.event_type}: {event.src_path}")

    def tick(self) -> None:
        if self._dirty and (time.time() - self._last_event_at) > DEBOUNCE_SECONDS:
            self._dirty = False
            print("\n[WATCH] 変更を検出。インデックスを更新します。")
            try:
                update_index()
            except Exception as e:
                print(f"[WATCH] 更新失敗: {e}")


def main() -> None:
    if not config.SOURCE_DIR.exists():
        raise FileNotFoundError(f"SOURCE_DIR が存在しません: {config.SOURCE_DIR}")

    print(f"[WATCH] 監視開始: {config.SOURCE_DIR}  (Ctrl+C で停止)")
    handler = IndexUpdater()
    observer = Observer()
    observer.schedule(handler, str(config.SOURCE_DIR), recursive=True)
    observer.start()
    try:
        while True:
            handler.tick()
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
        print("\n[WATCH] 停止しました。")
    observer.join()


if __name__ == "__main__":
    main()
