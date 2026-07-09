from fastapi import BackgroundTasks

from enterprise_rag.jobs.queue import FastApiBackgroundTaskQueue


class RecordingRunner:
    def __init__(self) -> None:
        self.ran = []

    def run(self, job_id: str) -> None:
        self.ran.append(job_id)


def test_fastapi_background_task_queue_publishes_job_to_background_tasks() -> None:
    background_tasks = BackgroundTasks()
    runner = RecordingRunner()

    FastApiBackgroundTaskQueue(background_tasks, runner).publish("job_123")
    for task in background_tasks.tasks:
        task.func(*task.args, **task.kwargs)

    assert runner.ran == ["job_123"]
