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

import argparse
import json
import multiprocessing as mp
import base64
import time
from tqdm import tqdm
from transformers import AutoTokenizer
import openai
from openai import OpenAI
from src.utils import *

# Global variable to store process-specific configuration
_process_data = None


def init_process(keys, base_url, data_path):
    """
    Initialize resources for each process in the pool.

    Each process:
    - Selects its own API key based on process ID
    - Creates an Azure OpenAI client instance
    - Loads the tokenizer for later use

    Args:
        keys (list[str]): List of API keys to be rotated per process.
        data_path (str): Path to the dataset file.
    """
    global _process_data

    # Assign API key based on process ID (rotating over the list)
    process_id = mp.current_process()._identity[0] if mp.current_process()._identity else 1
    key_index = (process_id - 1) % len(keys)
    key = keys[key_index]

    client = OpenAI(
        api_key=key,
        base_url=base_url,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct"
    )

    # Store process-local data
    _process_data = {
        'client': client,
        'data_path': data_path,
        "tokenizer": tokenizer
    }



def query_api(reference, prediction, question, client):
    """
    Query the Azure OpenAI API to rate the correctness of a model's prediction.

    The API is expected to return a single rating: '1', '2', or '3'.

    Args:
        reference (dict): Ground truth answer in structured format.
        prediction (str): Model's predicted answer.
        question (list[dict]): Question content (can be multimodal).
        client (openai.AzureOpenAI): API client.

    Returns:
        float: Correctness score mapped from rating to [0, 0.5, 1].
    """
    global _process_data
    client = _process_data['client']

    # Prepare multimodal input for API
    mm_input = [
        {"type": "text", "text": correctness_prompt},
        {"type": "text", "text": "\n<question>\n"}
    ] + question + [
        {"type": "text", "text": "\n</question>\n\n"},
        {"type": "text", "text": f"<reference_answer>\n{reference}\n</reference_answer>\n\n"},
        {"type": "text", "text": f"<assistant_answer>\n{prediction}\n</assistant_answer>\n\n"},
        {"type": "text", "text": (
            "Please respond with a single-word rating: '1', '2' or '3' without any explanation.\n"
            "Your rating:"
        )}
    ]

    # Convert local images to base64 before sending
    for idx in range(len(mm_input)):
        if mm_input[idx]['type'] == "image":
            mm_input[idx] = {
                "type": "image_url",
                "image_url": {
                    "url": image_to_base64(mm_input[idx]['image'], max_pixels=1280 * 28 * 28)
                }
            }

    max_retries = 120
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4.1-2025-04-14",
                messages=[{"role": "user", "content": mm_input}],
                max_tokens=20,
                temperature=0,
                top_p=1,
                n=1,
                timeout=3600,
            )

            result = response.choices[0].message.content
            if "1" in result:
                return 0
            elif "2" in result:
                return 0.5
            elif "3" in result:
                return 1
            return 0  # Fallback

        except Exception as e:
            if (attempt + 1) % 2 == 0:
                print(f"Process {mp.current_process().name} failed attempt [{attempt+1}/{max_retries}]: {e}")
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"Process {mp.current_process().name} failed after {max_retries} retries: {e}")
                return 0


def compute_correct_single_item(item_with_index):
    """
    Evaluate correctness for a single dataset item.

    Args:
        item_with_index (tuple): (index, item) where item is a dict with keys:
            'ground_truth', 'model_predictions', 'question'.

    Returns:
        dict: {'id': index, 'correctness': score}
    """
    global _process_data
    data_path = _process_data['data_path']
    tokenizer = _process_data['tokenizer']

    index, item = item_with_index

    # Clean model output depending on source type
    if ("think" in data_path.lower() and "glm" in data_path.lower()) or "mimo" in data_path.lower():
        if "</think>" in item['model_predictions']:
            ind = item['model_predictions'].index("</think>") + len("</think>")
            item['model_predictions'] = item['model_predictions'][ind:]
        else:
            pred_ids = tokenizer.encode(item['model_predictions'][-512:], add_special_tokens=False)
            item['model_predictions'] = tokenizer.decode(pred_ids)

    elif "think" in data_path.lower() and "kimi" in data_path.lower():
        if "◁/think▷" in item['model_predictions']:
            ind = item['model_predictions'].index("◁/think▷") + len("◁/think▷")
            item['model_predictions'] = item['model_predictions'][ind:]
        else:
            pred_ids = tokenizer.encode(item['model_predictions'][-512:], add_special_tokens=False)
            item['model_predictions'] = tokenizer.decode(pred_ids)

    # Format ground truth
    if isinstance(item['ground_truth'], list):
        ground_truth_lst = [f"{i+1}. {ans}\n" for i, ans in enumerate(item['ground_truth'])]
        ground_truth = {"type": 'text', "text": "\n".join(ground_truth_lst)}
    else:
        ground_truth = {"type": 'text', "text": item['ground_truth']}

    # Get correctness score from API
    score = query_api(ground_truth, item['model_predictions'], item['question'], _process_data['client'])

    return {"id": index, "correctness": score}


def main():
    """
    Main entry point for correctness evaluation.

    Loads a dataset file, evaluates each sample in parallel, and saves results to disk.
    """
    parser = argparse.ArgumentParser(description="Evaluate correctness of model predictions.")
    parser.add_argument("--file", type=str, required=True, help="Path to dataset JSON file.")
    parser.add_argument("--api_base_url", type=str, required=True)
    parser.add_argument("--api_keys", nargs="+", required=True, help="List of API KEYs")
    parser.add_argument("--processes", type=int, default=None, help="Number of worker processes (default: CPU cores x #keys).")
    args = parser.parse_args()

    # API keys list (replace 'xxx' with real keys)
    key_lst = args.api_keys

    with open(args.file, 'r') as f:
        data = json.load(f)

    file_name = args.file[:args.file.rindex('.')]

    # Determine number of processes to spawn
    num_processes = len(key_lst) * 4 if args.processes is None else args.processes
    print(f"Using {num_processes} processes to handle {len(data)} items")

    # Attach index to track original order
    data_with_index = list(enumerate(data))
    results = []

    # Process data in batches to avoid memory spikes
    for batch_st in range(0, len(data), 64):
        with mp.Pool(
            processes=num_processes,
            initializer=init_process,
            initargs=(key_lst, args.api_base_url, args.file)
        ) as pool:
            for result in tqdm(
                pool.imap_unordered(compute_correct_single_item, data_with_index[batch_st:batch_st+64]),
                total=len(data_with_index[batch_st:batch_st+64]),
                desc="Processing"
            ):
                results.append(result)

    # Restore original order after unordered mapping
    results = sorted(results, key=lambda x: x['id'])

    # Compute overall correctness
    result_dic = {
        "correctness": 100 * sum(r["correctness"] for r in results) / len(results)
    }

    # Add individual scores back into dataset
    for i in range(len(data)):
        data[i]['correct_score'] = results[i]["correctness"]

    # Save aggregated stats
    with open(file_name + "_correctness_result.json", "w") as f:
        json.dump(result_dic, f, indent=4)

    print("Processing complete!")
    print(f"Overall correctness: {result_dic['correctness']:.2f}%")


if __name__ == "__main__":
    main()
