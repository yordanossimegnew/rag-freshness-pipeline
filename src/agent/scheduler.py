"""Background scheduler: every tick, sync each due source (no overlapping syncs per source)."""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from ..utils import get_logger
from .freshness import evaluate_alerts
from .sync import Components, run_sync

log = get_logger("scheduler")


def run_claimed(parts: Components, source_id: str, trigger: str) -> dict:
    """Run a sync for a source the caller has already claimed, then re-evaluate alerts."""
    try:
        return run_sync(parts, source_id, trigger)
    finally:
        evaluate_alerts(parts.state, parts.settings.failure_alert_threshold, parts.settings.alert_webhook_url)


def sync_now(parts: Components, source_id: str, trigger: str = "manual") -> dict | None:
    """Claim the source and run one sync. None if a sync is already running."""
    if not parts.state.claim_source(source_id):
        return None
    return run_claimed(parts, source_id, trigger)


class Scheduler:
    def __init__(self, parts: Components):
        self.parts = parts
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="sync")

    def tick(self) -> int:
        due = [s for s in self.parts.state.due_sources() if self.parts.state.claim_source(s["id"])]
        for src in due:
            self._pool.submit(self._safe_sync, src["id"])
        if not due:  # SLAs keep ageing between syncs, so alerts are re-checked every tick
            evaluate_alerts(self.parts.state, self.parts.settings.failure_alert_threshold,
                            self.parts.settings.alert_webhook_url)
        return len(due)

    def _safe_sync(self, source_id: str) -> None:
        try:
            run_claimed(self.parts, source_id, "schedule")
        except Exception:  # noqa: BLE001
            log.exception("scheduled sync of %s failed", source_id)

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.tick()
            except Exception:  # noqa: BLE001 - e.g. database briefly unavailable
                log.exception("scheduler tick failed")
            self._stop.wait(self.parts.settings.scheduler_tick_seconds)

    def start(self) -> None:
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, name="scheduler", daemon=True)
            self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._pool.shutdown(wait=False, cancel_futures=True)
