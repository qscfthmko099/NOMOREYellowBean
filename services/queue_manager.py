import asyncio
from typing import Optional
from config.group_config import MAX_QUEUE_SIZE, MAX_CONCURRENT
from utils.logger import logger

class QueueFullError(Exception):
    """自定义队列已满异常"""

class MessageQueue:
    def __init__(self):
        self.queue = asyncio.Queue(maxsize=MAX_QUEUE_SIZE)
        self.concurrency = MAX_CONCURRENT
        self._workers = set()

    async def worker(self):
        while True:
            try:
                task = await self.queue.get()
                await task.process()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"任务处理失败: {str(e)}", exc_info=True)
            finally:
                self.queue.task_done()

    async def start(self):
        for _ in range(self.concurrency):
            worker = asyncio.create_task(self.worker())
            self._workers.add(worker)
            worker.add_done_callback(self._workers.remove)

    async def put(self, task):
        try:
            self.queue.put_nowait(task)
        except asyncio.QueueFull:
            logger.warning(f"消息队列已满（当前大小: {self.queue.qsize()}）")
            raise QueueFullError("消息队列已满，请稍后重试")

message_queue = MessageQueue()