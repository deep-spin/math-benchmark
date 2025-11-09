import asyncio
import argparse
import json
import os
from typing import Iterable, Set, Any, Dict
from openai import AsyncOpenAI, APITimeoutError
import tenacity
from tqdm import tqdm 

def iter_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    """Yield JSON objects from a JSONL file, skipping invalid/blank lines."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                # Skip malformed lines
                continue


def load_processed_ids(path: str) -> Set[Any]:
    """Collect set of already-processed ids from an existing JSONL file."""
    ids: Set[Any] = set()
    for obj in iter_jsonl(path) or []:
        qid = obj.get("id") if isinstance(obj, dict) else None
        if qid is not None:
            ids.add(qid)
    return ids


def append_jsonl(path: str, obj: Dict[str, Any]) -> None:
    """Append a single JSON object as one line to a JSONL file and fsync for durability."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    data = (json.dumps(obj, ensure_ascii=False) + "\n").encode("utf-8")
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.write(fd, data)
        os.fsync(fd)
    finally:
        os.close(fd)

class ChatModel:
    def __init__(self, model_name, base_url, api_key, timeout):
        # Create the OpenAI async client
        self.client = AsyncOpenAI(
            base_url=base_url,
            api_key=api_key,
            timeout=timeout,
        )
        self.model_name = model_name

    # Retry if not an APITimeoutError, with exponential backoff and jitter
    @tenacity.retry(
        wait=tenacity.wait_exponential(multiplier=1, min=4, max=10)
            + tenacity.wait_random(0, 2),  # small jitter to avoid thundering herd
        retry=tenacity.retry_if_exception(
            lambda exc: not isinstance(exc, APITimeoutError)
        ),
        stop=tenacity.stop_after_attempt(3),
        reraise=True,
    )
    async def __call__(self, question_text: str) -> dict:
        """Send messages to the chat model and return the response text."""
        completion = await self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                {"role": "user", "content": question_text}
            ],
        )
        choice = completion.choices[0]
        message = choice.message
        content = getattr(message, "content", None)
        if content is not None:
            return {
                "choice.message.content": content,
                "finish_reason": getattr(choice, "finish_reason", None),
                "model": getattr(completion, "model", None),
                "usage": completion.usage.model_dump() if getattr(completion, "usage", None) else None,
            }
        else:
            raise ValueError("Received empty content from API")


async def main():
    parser = argparse.ArgumentParser(description="Send prompts to a model server and store responses.")
    parser.add_argument(
        "--base_url",
        required=True,
        help="Base URL of the model server (e.g., http://localhost:8000/v1)",
    )
    parser.add_argument(
        "--api_key",
        required=True,
        help="API key for authentication with the model server",
    )
    parser.add_argument(
        "--prompts_path",
        required=True,
        help="Path to the prompts JSON file (produced by create_prompts.py)",
    )
    parser.add_argument(
        "--responses_path",
        required=True,
        help="Path to write the responses JSONL file (one JSON object per line; appended-to if exists; resume supported)",
    )
    parser.add_argument(
        "--model",
        default="/mnt/d/models/Qwen3-0.6B",
        help="Model name or path to use for chat completions",
    )
    parser.add_argument(
        "--concurrency", 
        type=int, 
        default=8,
        help="Number of concurrent requests to send",
    )
    args = parser.parse_args()
    # Create a chat model instance
    chat = ChatModel(model_name=args.model, base_url=args.base_url, api_key=args.api_key, timeout=1800)

    # Get list of prompts from the provided JSON file
    with open(args.prompts_path, "r", encoding="utf-8") as f:
        prompts = json.load(f)
    
    # JSONL output with resume: load processed IDs to skip already done items
    processed_ids = load_processed_ids(args.responses_path)

    def needs_processing(entry: Any) -> bool:
        qid = entry.get("id")
        return qid not in processed_ids

    pending = [e for e in prompts if needs_processing(e)]

    # True concurrency: limit in-flight requests with a semaphore and
    # serialize file writes with an asyncio.Lock to avoid interleaved output.
    sem = asyncio.Semaphore(max(1, args.concurrency))
    write_lock = asyncio.Lock()

    async def process(entry: Dict[str, Any]):
        async with sem:
            qid = entry.get("id")
            prompt_text = entry["prompt"]
            try:
                raw_response = await chat(prompt_text)
                record = {"id": qid, "raw_response": raw_response}
            except APITimeoutError as e:
                record = {"id": qid, "error": f"APITimeoutError: {str(e)}"}
            except Exception as e:
                record = {"id": qid, "error": f"Exception: {str(e)}"}
            # Ensure appends are serialized; each append is one write but guard for orderliness
            async with write_lock:
                append_jsonl(args.responses_path, record)

    # Create tasks for all pending entries and advance progress as they complete
    tasks = [asyncio.create_task(process(e)) for e in pending]
    with tqdm(total=len(pending), desc="Sending prompts", unit="prompt") as pbar:
        for fut in asyncio.as_completed(tasks):
            try:
                await fut
            finally:
                pbar.update(1)

if __name__ == "__main__":
    asyncio.run(main())
