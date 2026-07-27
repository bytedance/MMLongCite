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
import threading
import base64
import time
from tqdm import tqdm
from transformers import AutoTokenizer
import openai
from openai import OpenAI
from utils import *
from concurrent.futures import ThreadPoolExecutor, as_completed
import os

# Thread-local storage for thread-specific configuration
_thread_data = threading.local()

# Global configuration shared across threads
_global_config = {}
_thread_counter = 0
_counter_lock = threading.Lock()


def get_thread_id():
    """
    Get a unique sequential ID for each thread.
    
    Returns:
        int: Unique thread ID starting from 1.
    """
    global _thread_counter
    
    if not hasattr(_thread_data, 'thread_id'):
        with _counter_lock:
            _thread_counter += 1
            _thread_data.thread_id = _thread_counter
    
    return _thread_data.thread_id


def init_thread():
    """
    Initialize resources for the current thread.
    
    Each thread:
    - Selects its own API key based on thread ID
    - Creates an OpenAI client instance
    - Loads the tokenizer for later use
    """
    if hasattr(_thread_data, 'client'):
        # Already initialized
        return
    
    keys = _global_config['keys']
    base_url = _global_config['base_url']
    data_path = _global_config['data_path']
    
    # Assign API key based on thread ID (rotating over the list)
    thread_id = get_thread_id()
    key_index = (thread_id - 1) % len(keys)
    key = keys[key_index]
    
    client = OpenAI(
        api_key=key,
        base_url=base_url,
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct"
    )
    
    # Store thread-local data
    _thread_data.client = client
    _thread_data.data_path = data_path
    _thread_data.tokenizer = tokenizer


def query_api(reference, prediction, question, client):
    """
    Query the OpenAI API to rate the correctness of a model's prediction.
    
    The API is expected to return a single rating: '1', '2', or '3'.
    
    Args:
        reference (dict): Ground truth answer in structured format.
        prediction (str): Model's predicted answer.
        question (list[dict]): Question content (can be multimodal).
        client (openai.OpenAI): API client.
    
    Returns:
        float: Correctness score mapped from rating to [0, 0.5, 1].
    """
    init_thread()
    client = _thread_data.client
    
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
                    "url": image_to_base64(mm_input[idx]['image'], max_pixels=1024 * 28 * 28)
                }
            }
    
    max_retries = 30
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-5.6-sol",
                messages=[{"role": "user", "content": mm_input}],
                max_tokens=1024,
                temperature=0,
                top_p=1,
                n=1,
                timeout=60,
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
                print(f"Thread {threading.current_thread().name} failed attempt [{attempt+1}/{max_retries}]: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                print(f"Thread {threading.current_thread().name} failed after {max_retries} retries: {e}")
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
    init_thread()
    data_path = _thread_data.data_path
    tokenizer = _thread_data.tokenizer
    
    index, item = item_with_index
    
    # Clean model output depending on source type
    if ("think" in data_path.lower() and "glm" in data_path.lower()) or "mimo" in data_path.lower():
        if "</think>" in item['model_predictions']:
            ind = item['model_predictions'].rindex("</think>") + len("</think>")
            item['model_predictions'] = item['model_predictions'][ind:]
            pred_ids = tokenizer.encode(item['model_predictions'], add_special_tokens=False)[:128]
            item['model_predictions'] = tokenizer.decode(pred_ids)
        else:
            pred_ids = tokenizer.encode(item['model_predictions'], add_special_tokens=False)[-128:]
            item['model_predictions'] = tokenizer.decode(pred_ids)
    
    elif "think" in data_path.lower() and "kimi" in data_path.lower():
        if "◁/think▷" in item['model_predictions']:
            ind = item['model_predictions'].rindex("◁/think▷") + len("◁/think▷")
            item['model_predictions'] = item['model_predictions'][ind:]
            pred_ids = tokenizer.encode(item['model_predictions'], add_special_tokens=False)[:128]
            item['model_predictions'] = tokenizer.decode(pred_ids)
        else:
            pred_ids = tokenizer.encode(item['model_predictions'], add_special_tokens=False)[-128:]
            item['model_predictions'] = tokenizer.decode(pred_ids)
    
    # Format ground truth
    if isinstance(item['ground_truth'], list):
        ground_truth_lst = [f"{i+1}. {ans}\n" for i, ans in enumerate(item['ground_truth'])]
        ground_truth = {"type": 'text', "text": "\n".join(ground_truth_lst)}
    else:
        ground_truth = {"type": 'text', "text": item['ground_truth']}
    
    # Get correctness score from API
    score = query_api(ground_truth, item['model_predictions'], item['question'], _thread_data.client)
    
    return {"id": index, "correctness": score}


def main():
    """
    Main entry point for correctness evaluation.
    
    Loads a dataset file, evaluates each sample in parallel using threads, and saves results to disk.
    """
    global _global_config, _thread_counter
    
    parser = argparse.ArgumentParser(description="Evaluate correctness of model predictions.")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--api_base_url", type=str, required=True)
    parser.add_argument("--api_keys", nargs="+", required=True, help="List of API KEYs")
    parser.add_argument("--threads", type=int, default=None, help="Number of worker threads (default: #keys x 4).")
    args = parser.parse_args()
    
    # API keys list
    key_lst = args.api_keys
    model_name = args.model
    task = args.task
    file_path = os.path.join("results", task, model_name + ".json")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Determine number of threads to spawn
    num_threads = len(key_lst) * 4 if args.threads is None else args.threads
    print(f"Using {num_threads} threads to handle {len(data)} items")
    
    # Set global configuration
    _global_config = {
        'keys': key_lst,
        'base_url': args.api_base_url,
        'data_path': file_path
    }
    
    # Attach index to track original order
    data_with_index = list(enumerate(data))
    results = []
    
    # Process data in batches to avoid memory spikes
    for batch_st in range(0, len(data), 1024):
        batch_data = data_with_index[batch_st:batch_st+1024]
        
        # Reset thread counter for each batch
        _thread_counter = 0
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks
            futures = {executor.submit(compute_correct_single_item, item): item for item in batch_data}
            
            # Process completed tasks with progress bar
            for future in tqdm(as_completed(futures), total=len(batch_data), desc="Processing"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Task failed with error: {e}")
                    # Append a default result to maintain data integrity
                    item_index = futures[future][0]
                    results.append({"id": item_index, "correctness": 0})
    
    # Restore original order after unordered mapping
    results = sorted(results, key=lambda x: x['id'])
    for i in range(len(data)):
        data[i]['correctness'] = results[i]["correctness"]
    with open(os.path.join("results", task, model_name + "_correctness_result.json"), "w") as f:
        json.dump(data, f, indent=4)
    
    # Compute overall correctness
    result_dic = {
        "correctness": 100 * sum(r["correctness"] for r in results) / len(results)
    }
    
    # Add individual scores back into dataset
    for i in range(len(data)):
        data[i]['correct_score'] = results[i]["correctness"]
    
    # Save aggregated stats
    with open(os.path.join("results", task, model_name + "_correctness_score.json"), "w") as f:
        json.dump(result_dic, f, indent=4)
    
    print("Processing complete!")
    print(f"Overall correctness: {result_dic['correctness']:.2f}%")


if __name__ == "__main__":
    main()
