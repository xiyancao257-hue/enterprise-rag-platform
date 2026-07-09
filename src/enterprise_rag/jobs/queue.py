from __future__ import annotations

from typing import Protocol

from fastapi import BackgroundTasks

from enterprise_rag.jobs.runner import IngestJobRunner


class IngestJobQueue(Protocol):
    def publish(self, job_id: str) -> None:
        """Publish an ingest job for asynchronous execution."""


class FastApiBackgroundTaskQueue:
    def __init__(self, background_tasks: BackgroundTasks, runner: IngestJobRunner) -> None:
        self.background_tasks = background_tasks
        self.runner = runner

    def publish(self, job_id: str) -> None:
        self.background_tasks.add_task(self.runner.run, job_id)
