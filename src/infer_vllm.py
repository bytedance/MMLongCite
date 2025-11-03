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

import os
import json
import base64
import torch
import argparse
from tqdm import tqdm
from vllm import LLM, SamplingParams
from transformers import AutoProcessor
from vllm.multimodal.utils import fetch_image
from qwen_vl_utils import process_vision_info
from src.utils import *

def to_base64(path):
    """
    Convert a file (image/video) to a Base64-encoded string with proper MIME type.

    Args:
        path (str): Path to the file.

    Returns:
        str: Base64 string with data URI prefix.
    """
    with open(path, 'rb') as file:
        binary_data = file.read()
        base64_string = base64.b64encode(binary_data).decode('utf-8')

    if path.endswith(".jpg"):
        return f"data:image/jpeg;base64,{base64_string}"
    elif path.endswith(".png"):
        return f"data:image/png;base64,{base64_string}"
    elif path.endswith(".mp4"):
        return f"data:video/mp4;base64,{base64_string}"
    else:
        raise ValueError(f"Unsupported file type: {path}")


def preprocess(data, task, min_pixels=None, max_pixels=None):
    """
    Preprocess dataset samples into a chat message format suitable for multimodal LLM input.

    Args:
        data (list): List of dataset items. Each contains 'context' and 'question'.
        task (str): Task name or dataset identifier.
        min_pixels (int, optional): Minimum resolution for images. (Currently unused)
        max_pixels (int, optional): Maximum resolution for images. (Currently unused)

    Returns:
        list: Processed data in structured message format.
    """
    processed_data = []

    for item in tqdm(data, total=len(data), desc="Processing Data"):
        context = item.get("context", [])

        # Initialize messages list (one "user" message containing all context + instruction + question).
        messages = [{"role": "user", "content": []}]

        # Process each context entry
        for idx in range(len(context)):
            is_video_task = "video" in task.lower()

            if context[idx]['type'] == "image":
                label = "Frame" if is_video_task else "Image"
                messages[0]["content"].append({"type": "text", "text": f"{label} [{idx + 1}]: "})
                messages[0]["content"].append({"type": "image", "image": to_base64(context[idx]['image'])})

            elif context[idx]['type'] == "text":
                messages[0]["content"].append({
                    "type": "text",
                    "text": f"Passage [{idx + 1}]: {context[idx]['text']}"
                })

            # Separator between items
            messages[0]["content"].append({"type": "text", "text": "\n"})

        # Add task-specific instruction and the question prompt
        messages[0]["content"].append({
            "type": "text",
            "text": f"\n\n{Instruction[task]}\n\nQuestion:\n"
        })

        # Append question content (image or text)
        for q_item in item['question']:
            if q_item['type'] == "image":
                messages[0]["content"].append({"type": "image", "image": to_base64(q_item['image'])})
            else:
                messages[0]["content"].append({"type": "text", "text": q_item['text']})

        # Mark position for the model's answer
        messages[0]["content"].append({"type": "text", "text": "\nAnswer:\n"})

        processed_data.append(messages)

    return processed_data


def main(args):
    """
    Main inference workflow:
    - Load model and processor
    - Prepare data
    - Generate predictions
    - Save results to JSON
    """
    model_name = args.model
    model_path = MODEL_DIC[model_name]

    # Initialize LLM
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=0.9,
        tensor_parallel_size=torch.cuda.device_count(),
        dtype="auto",
        trust_remote_code=True
    )

    # Initialize processor for chat template formatting
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

    # Configure sampling parameters
    sampling_params = SamplingParams(
        temperature=model_generation_params[model_name]['temperature'],
        top_p=model_generation_params[model_name]['top_p'],
        max_tokens=model_generation_params[model_name]['max_tokens'],
    )

    # Allow more tokens if "thinking" mode is enabled
    if args.thinking:
        sampling_params.max_tokens = 32768

    for dataset in args.dataset:
        # Load dataset from JSON file
        with open(DATASET_INFO[dataset]['file_path'], 'r') as f:
            data = json.load(f)

        # Preprocess input data
        processed_data = preprocess(data, dataset)

        # Prepare inputs for batched inference
        llm_inputs_batch = []
        for messages in tqdm(processed_data, desc=f"Preparing batch for Dataset: {dataset}"):
            # Format chat prompt
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            # Extract multimodal input data
            if "qwen" in model_name.lower():
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                image_inputs = [
                    fetch_image(item['image'])
                    for item in messages[0]["content"] if item['type'] == 'image'
                ]
                video_inputs = None

            mm_data = {}
            if image_inputs:
                mm_data["image"] = image_inputs
            if video_inputs:
                mm_data["video"] = video_inputs

            llm_inputs_batch.append({
                "prompt": prompt,
                "multi_modal_data": mm_data,
            })

        # Perform batched inference (in chunks of 1000 samples)
        outputs = []
        for batch_start in range(0, len(llm_inputs_batch), 1000):
            batch_end = batch_start + 1000
            outputs.extend(llm.generate(
                llm_inputs_batch[batch_start:batch_end],
                sampling_params=sampling_params
            ))

        # Collect responses
        responses = [output.outputs[0].text for output in outputs]

        # Append predictions to original data
        results = []
        for i, resp in enumerate(responses):
            data[i]["model_predictions"] = resp
            results.append(data[i])

        # Ensure target directory exists
        save_path = os.path.join("results", dataset, f'{model_name}.json')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save as JSON
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multimodal LLM inference script")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--dataset", nargs="+", required=True, help="List of dataset names")
    parser.add_argument("--thinking", action='store_true', default=False, help="Enable extended reasoning mode")
    args = parser.parse_args()

    main(args)
