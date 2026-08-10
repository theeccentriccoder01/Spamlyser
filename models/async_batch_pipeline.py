"""
Asynchronous Batch Email Spam Inference Pipeline & Concurrent Processing Queue.
Provides high-throughput async processing for bulk spam detection with concurrency control,
retry strategy, and batch payload aggregation.
"""

import asyncio
from typing import List, Dict, Any, Callable, Optional
import time
import logging

logger = logging.getLogger(__name__)

class AsyncBatchPipeline:
    """
    High-throughput async batch processing pipeline for email spam classification.
    """

    def __init__(
        self,
        classifier_func: Callable[[str], Dict[str, Any]],
        max_concurrency: int = 5,
        batch_size: int = 50,
        max_retries: int = 2
    ):
        self.classifier_func = classifier_func
        self.max_concurrency = max_concurrency
        self.batch_size = batch_size
        self.max_retries = max_retries
        self.semaphore = asyncio.Semaphore(max_concurrency)

    async def _process_single(self, text: str, item_id: int) -> Dict[str, Any]:
        async with self.semaphore:
            attempts = 0
            while attempts <= self.max_retries:
                try:
                    start_time = time.perf_counter()
                    # Execute synchronous classification function in thread pool executor
                    loop = asyncio.get_running_loop()
                    res = await loop.run_in_executor(None, self.classifier_func, text)
                    latency = (time.perf_counter() - start_time) * 1000.0

                    return {
                        "id": item_id,
                        "status": "success",
                        "result": res,
                        "latency_ms": round(latency, 2),
                        "attempts": attempts + 1
                    }
                except Exception as e:
                    attempts += 1
                    logger.warning(f"Error processing item {item_id} (Attempt {attempts}): {e}")
                    if attempts > self.max_retries:
                        return {
                            "id": item_id,
                            "status": "failed",
                            "error": str(e),
                            "attempts": attempts
                        }
                    await asyncio.sleep(0.05 * attempts)

    async def process_batch(self, payload: List[str]) -> Dict[str, Any]:
        """
        Process a list of input message strings concurrently.
        """
        start_batch = time.perf_counter()
        tasks = [
            self._process_single(text, idx)
            for idx, text in enumerate(payload)
        ]
        results = await asyncio.gather(*tasks)

        total_time = (time.perf_counter() - start_batch) * 1000.0
        successes = [r for r in results if r["status"] == "success"]
        failures = [r for r in results if r["status"] == "failed"]

        return {
            "total_processed": len(payload),
            "successful": len(successes),
            "failed": len(failures),
            "total_latency_ms": round(total_time, 2),
            "avg_latency_per_item_ms": round(total_time / len(payload), 2) if payload else 0.0,
            "results": results
        }

def run_async_batch(classifier_func: Callable[[str], Dict[str, Any]], payload: List[str], max_concurrency: int = 5) -> Dict[str, Any]:
    """
    Synchronous wrapper to execute the async pipeline.
    """
    pipeline = AsyncBatchPipeline(classifier_func=classifier_func, max_concurrency=max_concurrency)
    return asyncio.run(pipeline.process_batch(payload))
