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
import collections
import json
import re
import string
import copy
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import requests
import time
import os
import sys

from functools import partial
from nltk import sent_tokenize
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer
import openai
from openai import OpenAI
from utils import *

# ============================================================
# Thread-local storage
# ============================================================
_thread_local = threading.local()
_global_config = {}
_thread_id_counter = 0
_thread_id_lock = threading.Lock()


def get_thread_id():
    """
    Get a unique thread ID for the current thread.
    """
    global _thread_id_counter
    if not hasattr(_thread_local, 'thread_id'):
        with _thread_id_lock:
            _thread_local.thread_id = _thread_id_counter
            _thread_id_counter += 1
    return _thread_local.thread_id


def init_thread_data():
    """
    Initialize per-thread configurations:
    - Select the API key based on thread ID.
    - Create an OpenAI client instance for each thread.
    - Load tokenizer for text processing (shared across threads).
    """
    if hasattr(_thread_local, 'initialized') and _thread_local.initialized:
        return
    
    thread_id = get_thread_id()
    keys = _global_config['keys']
    base_url = _global_config['base_url']
    
    key_index = thread_id % len(keys)
    key = keys[key_index]
    
    client = OpenAI(
        api_key=key,
        base_url=base_url,
    )
    
    _thread_local.client = client
    _thread_local.data_path = _global_config['data_path']
    _thread_local.at_most_citations = _global_config['at_most_citations']
    _thread_local.tokenizer = _global_config['tokenizer']
    _thread_local.initialized = True


def remove_citations(sent):
    """
    Remove citation markers like [1], [23] from sentence strings.
    """
    return re.sub(r"\[\d+", "", re.sub(r" \[\d+", "", sent)).replace(" |", "").replace("]", "")


def _run_nli_autoais(passage, claim, question):
    """
    Perform NLI (Natural Language Inference) using the initialized OpenAI client
    to determine if the claim is relevant to the given passage and question.
    Returns:
        1 if relevant, 0 if irrelevant.
    """
    init_thread_data()
    client = _thread_local.client
    
    mm_input = [
        {"type": "text", "text": relevant_prompt},
        {"type": "text", "text": "\n<question>\n"}
    ] + question + [
        {"type": "text", "text": "\n</question>\n\n"},
        {"type": "text", "text": f"<statement>\n{claim}\n</statement>\n\n"},
        {"type": "text", "text": "<content>\n"}
    ] + passage + [
        {"type": "text", "text": "\n</content>\n\n"},
        {"type": "text", "text": "Please respond with a single-word rating: 'Relevant' or 'Irrelevant' without any explanation.\nYour rating:"}
    ]
    
    # Convert image input to base64
    for i, input_item in enumerate(mm_input):
        if input_item['type'] == "image":
            mm_input[i] = {
                "type": "image_url",
                "image_url": {
                    "url": image_to_base64(input_item['image'], max_pixels=1024 * 28 * 28)
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

            if "irrelevant" in result.lower():
                return 0
            elif "relevant" in result.lower():
                return 1
            return 0
        
        except Exception as e:
            if (attempt + 1) % 2 == 0:
                print(f"Thread {threading.current_thread().name} failed attempt [{attempt+1}/{max_retries}]: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                print(f"Thread {threading.current_thread().name} failed after {max_retries} retries: {str(e)}")
                return 0  # Default value


def _format_document(doc):
    """
    Format document content into standardized OpenAI input format.
    Handles text, image, and video frame types differently.
    """
    init_thread_data()
    data_path = _thread_local.data_path
    
    if doc['type'] == 'text':
        return [{"type": "text", "text": f"Passage: {doc['text']}\n"}]
    else:
        prefix = "Frame: " if "video" in data_path.lower() else "Image: "
        return [
            {"type": "text", "text": prefix},
            {"type": "image", "image": doc['image']},
            {"type": "text", "text": "\n"},
        ]


def compute_autoais_single_item(item_with_index, task):
    """
    Process a single dataset item for citation relevance evaluation.
    This function runs inside worker threads of ThreadPoolExecutor.
    """
    init_thread_data()
    at_most_citations = _thread_local.at_most_citations
    data_path = _thread_local.data_path
    tokenizer = _thread_local.tokenizer
    
    index, item = item_with_index
    
    if not item['model_predictions']:
        return {
            "id": index,
            "citation_rec": 0,
            "citation_prec": 0,
            "citation_f1": 0,
        }
    
    # Post-process model predictions depending on dataset source/type
    if "think" in data_path.lower() and ("glm" in data_path.lower() or "mimo" in data_path.lower() or "qwen" in data_path.lower()):
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
    
    # Sentence segmentation
    sents = sent_tokenize(item['model_predictions'])
    if len(sents) == 0:
        return {
            "id": index,
            "citation_rec": 0,
            "citation_prec": 0,
            "citation_f1": 0,
        }
        
    context = item['meta']['original_context'] if 'hr' in task else item['context']
    
    target_sents = [remove_citations(sent).strip() for sent in sents]
    
    entail = 0
    entail_prec = 0
    total_citations = 0

    
    # Iterate over each sentence for citation analysis
    for sent_id, sent in enumerate(sents):
        target_sent = target_sents[sent_id]
        joint_entail = -1
        
        ref = [int(r[1:]) - 1 for r in re.findall(r"\[\d+", sent)]
        if len(ref) == 0 or any([ref_id >= len(context) for ref_id in ref]):
            joint_entail = 0
        else:
            if at_most_citations is not None:
                ref = ref[:at_most_citations]
            total_citations += len(ref)
            joint_passage = []
            for psgs_id in ref:
                joint_passage += _format_document(context[psgs_id])
        
        if joint_entail == -1:
            joint_entail = _run_nli_autoais(joint_passage, target_sent, item['question'])
        
        entail += joint_entail
        
        if joint_entail and len(ref) > 1:
            for psgs_id in ref:
                passage = _format_document(context[psgs_id])
                nli_result = _run_nli_autoais(passage, target_sent, item['question'])
                
                if not nli_result:
                    subset_exclude = copy.deepcopy(ref)
                    subset_exclude.remove(psgs_id)
                    passage = []
                    for pid in subset_exclude:
                        passage += _format_document(context[pid])
                    nli_result = _run_nli_autoais(passage, target_sent, item['question'])
                    if not nli_result:
                        entail_prec += 1
                else:
                    entail_prec += 1
        else:
            entail_prec += joint_entail
    
    recall = entail / len(sents) if len(sents) > 0 else 0
    precision = entail_prec / total_citations if total_citations > 0 else 0
    f1 = (2 * precision * recall) / (precision + recall) if precision + recall != 0 else 0
    
    return {
        "id": index,
        "citation_rec": recall,
        "citation_prec": precision,
        "citation_f1": f1,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--at_most_citations", type=int, default=3)
    parser.add_argument("--task", type=str, required=True)
    parser.add_argument("--api_base_url", type=str, required=True)
    parser.add_argument("--api_keys", nargs="+", required=True, help="List of API KEYs")
    parser.add_argument("--threads", type=int, default=None, help="Number of threads to use (default: num_keys * 4)")
    
    args = parser.parse_args()
    
    key_lst = args.api_keys
    file_path = os.path.join("results", args.task, args.model + ".json")
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    model_name = args.model
    task = args.task
    
    num_threads = len(key_lst) * 4 if args.threads is None else args.threads
    print(f"Using {num_threads} threads to handle {len(data)} items.")
    
    # Initialize global configuration (shared across all threads)
    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-VL-7B-Instruct"
    )
    
    _global_config['keys'] = key_lst
    _global_config['base_url'] = args.api_base_url
    _global_config['data_path'] = file_path
    _global_config['at_most_citations'] = args.at_most_citations
    _global_config['tokenizer'] = tokenizer
    
    data_with_index = list(enumerate(data))
    results = []
    
    # Process data in batches of 64
    for batch_start in range(0, len(data), 1024):
        batch_data = data_with_index[batch_start:batch_start + 1024]
        
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            # Submit all tasks
            futures = {executor.submit(compute_autoais_single_item, item, task): item for item in batch_data}
            
            # Process results as they complete with progress bar
            for future in tqdm(as_completed(futures), total=len(batch_data), desc="Processing"):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    print(f"Error processing item: {e}")
    
    results = sorted(results, key=lambda x: x['id'])
    for i in range(len(data)):
        data[i]['citation_precision'] = results[i]["citation_prec"]
        data[i]['citation_recall'] = results[i]["citation_rec"]
        data[i]['citation_f1'] = results[i]["citation_f1"]
    with open(os.path.join("results", task, model_name + "_citation_result.json"), "w") as f:
        json.dump(data, f, indent=4)
    
    result_dic = {
        "citation_rec": 100 * sum(r["citation_rec"] for r in results) / len(results),
        "citation_prec": 100 * sum(r["citation_prec"] for r in results) / len(results),
        "citation_f1": 100 * sum(r["citation_f1"] for r in results) / len(results),
    }
    
    for i, item in enumerate(data):
        item['citation_recall'] = results[i]["citation_rec"]
        item['citation_precision'] = results[i]["citation_prec"]
        item['citation_f1'] = results[i]["citation_f1"]
    
    with open(os.path.join("results", task, model_name + "_citation_score.json"), "w") as f:
        json.dump(result_dic, f, indent=4)
    
    print("Processing completed!")
    print(f"Results - Recall: {result_dic['citation_rec']:.2f}%, Precision: {result_dic['citation_prec']:.2f}%, F1: {result_dic['citation_f1']:.2f}%")


if __name__ == "__main__":
    main()
