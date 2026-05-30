# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from vllm.multimodal.utils import fetch_image
from qwen_vl_utils import process_vision_info

from utils import (
    DATASET_INFO,
    Instruction,
    MODEL_DIC,
    image_to_base64,
    model_generation_params,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
# Max pixels budget for MM-NIAH (1024 patches of 14x14 with patch_size_factor=4)
MM_NIAH_MAX_PIXELS: int = 1024 * 14 * 14 * 4

# Inference micro-batch size (vLLM handles internal scheduling, but we chunk
# the inputs to avoid building one giant Python list / future regression).
INFERENCE_CHUNK_SIZE: int = 1000

# Max output tokens when --thinking is enabled.
THINKING_MAX_TOKENS: int = 32768

# Default GPU memory utilization for vLLM.
GPU_MEM_UTIL: float = 0.9

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format="[%(asctime)s] [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)



def _encode_image(image: Any, task: str) -> str:
    """Task-aware image encoder.

    For MM-NIAH we cap the resolution to keep token counts manageable.
    """
    if task == "mm-niah":
        return image_to_base64(image, min_pixels=None, max_pixels=MM_NIAH_MAX_PIXELS)
    return image_to_base64(image)


# ---------------------------------------------------------------------------
# Data preprocessing
# ---------------------------------------------------------------------------
def preprocess(data: List[Dict[str, Any]], task: str) -> List[List[Dict[str, Any]]]:
    """Convert dataset items into chat-style messages for the MLLM.

    Each item is assembled into a single "user" message that interleaves
    context (images / frames / passages), an instruction, and the question.

    Args:
        data: Raw dataset items. Each item must have `context` and `question`.
        task: Dataset / task name (used to pick the instruction template).

    Returns:
        A list of message lists, one per sample.
    """
    is_video_task = "video" in task.lower()
    label = "Frame" if is_video_task else "Image"
    processed: List[List[Dict[str, Any]]] = []

    for item in tqdm(data, desc=f"Preprocessing [{task}]"):
        content: List[Dict[str, Any]] = []

        # ---- Context block --------------------------------------------------
        for idx, ctx in enumerate(item.get("context", []), start=1):
            ctype = ctx["type"]
            if ctype == "image":
                if "hr" not in task:
                    content.append({"type": "text", "text": f"{label} [{idx}]: "})
                content.append({"type": "image", "image": _encode_image(ctx["image"], task)})
            elif ctype == "text":
                content.append({"type": "text", "text": f"Passage [{idx}]: {ctx['text']}"})
            else:
                raise ValueError(f"Unknown context type: {ctype}")
            content.append({"type": "text", "text": "\n"})

        # ---- Instruction + Question ----------------------------------------
        content.append({"type": "text", "text": f"\n\n{Instruction[task]}\n\nQuestion:\n"})
        for q in item["question"]:
            if q["type"] == "image":
                content.append({"type": "image", "image": _encode_image(q["image"], task)})
            else:
                content.append({"type": "text", "text": q["text"]})
        content.append({"type": "text", "text": "\nAnswer:\n"})

        processed.append([{"role": "user", "content": content}])

    return processed


# ---------------------------------------------------------------------------
# Per-sample LLM input preparation
# ---------------------------------------------------------------------------
def prepare_single(
    idx: int,
    messages: List[Dict[str, Any]],
    processor: AutoProcessor,
    model_name: str,
    thinking: bool,
) -> Tuple[int, Dict[str, Any]]:
    """Build the vLLM input dict for a single sample.

    Returns:
        (idx, llm_input) where `llm_input` has keys `prompt` and `multi_modal_data`.
    """
    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
    )

    # Extract multimodal payloads -- Qwen-VL has its own helper that handles
    # resizing / token-budget logic; for other models we just fetch raw images.
    if "qwen" in model_name.lower():
        image_inputs, video_inputs = process_vision_info(messages)
    else:
        image_inputs = [
            fetch_image(item["image"])
            for item in messages[0]["content"]
            if item["type"] == "image"
        ]
        video_inputs = None

    mm_data: Dict[str, Any] = {}
    if image_inputs:
        mm_data["image"] = image_inputs
    if video_inputs:
        mm_data["video"] = video_inputs

    return idx, {"prompt": prompt, "multi_modal_data": mm_data}


def prepare_batch_parallel(
    processed_data: List[List[Dict[str, Any]]],
    processor: AutoProcessor,
    model_name: str,
    thinking: bool,
    num_workers: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Prepare all LLM inputs using a thread pool.

    The workload here is I/O-bound (image decoding, base64 encoding), so
    threads outperform processes and avoid IPC serialization overhead.

    Args:
        processed_data: Output of `preprocess()`.
        processor:      Shared HF processor (read-only / thread-safe usage).
        model_name:     Used to dispatch vision-info extraction.
        thinking:       Whether extended-reasoning mode is enabled.
        num_workers:    Thread count. Defaults to `min(32, cpu_count * 2)`.

    Returns:
        A list of vLLM input dicts, in the same order as `processed_data`.
    """
    if num_workers is None:
        num_workers = min(32, (os.cpu_count() or 1) * 2)
    logger.info("Preparing %d samples with %d threads", len(processed_data), num_workers)

    results: List[Optional[Dict[str, Any]]] = [None] * len(processed_data)

    with ThreadPoolExecutor(max_workers=num_workers) as pool:
        futures = {
            pool.submit(prepare_single, i, msg, processor, model_name, thinking): i
            for i, msg in enumerate(processed_data)
        }
        with tqdm(total=len(futures), desc="Preparing batch") as pbar:
            for fut in as_completed(futures):
                try:
                    i, llm_input = fut.result()
                    results[i] = llm_input
                except Exception as e:
                    original_idx = futures[fut]
                    logger.error("Failed to prepare sample %d: %s", original_idx, e)
                    raise
                finally:
                    pbar.update(1)

    return results


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def build_llm(model_path: str) -> LLM:
    """Instantiate a vLLM engine with sensible defaults."""
    return LLM(
        model=model_path,
        gpu_memory_utilization=GPU_MEM_UTIL,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype="auto",
        trust_remote_code=True,
    )


def build_sampling_params(model_name: str, thinking: bool) -> SamplingParams:
    """Construct SamplingParams from the per-model config in `utils.py`."""
    params = model_generation_params[model_name]
    max_tokens = THINKING_MAX_TOKENS if thinking else params["max_tokens"]
    return SamplingParams(
        temperature=params["temperature"],
        top_p=params["top_p"],
        max_tokens=max_tokens,
    )


def run_dataset(
    dataset: str,
    llm: LLM,
    processor: AutoProcessor,
    sampling_params: SamplingParams,
    args: argparse.Namespace,
) -> None:
    """Run inference for a single dataset and write predictions to disk."""
    logger.info("=" * 60)
    logger.info("Running dataset: %s", dataset)
    logger.info("=" * 60)

    with open(DATASET_INFO[dataset]["file_path"], "r", encoding="utf-8") as f:
        data = json.load(f)

    processed = preprocess(data, dataset)
    llm_inputs = prepare_batch_parallel(
        processed,
        processor,
        args.model,
        args.thinking,
        num_workers=args.num_workers,
    )

    # Chunked generation.
    outputs = []
    for start in tqdm(
        range(0, len(llm_inputs), INFERENCE_CHUNK_SIZE),
        desc=f"Inference on {dataset}",
    ):
        chunk = llm_inputs[start : start + INFERENCE_CHUNK_SIZE]
        outputs.extend(llm.generate(chunk, sampling_params=sampling_params))

    # Merge predictions back into the original samples.
    for item, out in zip(data, outputs):
        item["model_predictions"] = out.outputs[0].text

    # Persist results.
    suffix = "_thinking" if args.thinking else ""
    save_path = Path("results") / dataset / f"{args.model}{suffix}.json"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info("Saved %d predictions to %s", len(data), save_path)


def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    if args.model not in MODEL_DIC:
        raise ValueError(
            f"Unknown model '{args.model}'. Available: {list(MODEL_DIC.keys())}"
        )
    unknown = [d for d in args.dataset if d not in DATASET_INFO]
    if unknown:
        raise ValueError(
            f"Unknown dataset(s): {unknown}. Available: {list(DATASET_INFO.keys())}"
        )

    model_path = MODEL_DIC[args.model]
    logger.info("Loading model: %s (%s)", args.model, model_path)

    llm = build_llm(model_path)
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    sampling_params = build_sampling_params(args.model, args.thinking)

    for dataset in args.dataset:
        run_dataset(dataset, llm, processor, sampling_params, args)



def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multimodal LLM inference script (vLLM backend).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help=f"Choices: {list(MODEL_DIC.keys())}",
    )
    parser.add_argument(
        "--dataset", nargs="+", required=True,
        help=f"Choices: {list(DATASET_INFO.keys())}",
    )
    parser.add_argument(
        "--thinking", action="store_true",
        help="Enable extended reasoning ('thinking') mode.",
    )
    parser.add_argument(
        "--num_workers", type=int, default=None,
        help="Threads for parallel preprocessing. Defaults to min(32, 2*cpu_count).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main(parse_args())
