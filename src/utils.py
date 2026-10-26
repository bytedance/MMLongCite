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

from PIL import Image
import io
import base64
import math


text_only_instruction = """Instruction
Please answer the question with citations based only on the provided passages. You must adhere to the following rules:
- Use an unbiased and journalistic tone.
- **Cite at least one and at most three passages at the end of each sentence in your response.**
- Cite passages using only their index number in brackets, for example [4][7], not [Passage 4][Passage 7].
- When citing several passages, use separate brackets for each index number, like [a][b][c], instead of combining them like [a, b, c].
- If multiple passages support a sentence, cite only a minimum sufficient subset of the passages.

----------------------
Output Format Example
Question:
What is the cause of death of director of film The Sap (1926 Film)?
Answer:
The director of the film The Sap (1926) was Erle C. Kenton [6][33]. He passed away on January 28, 1980, in Glendale, California, due to Parkinson's disease [17].
----------------------"""


image_only_instruction = """Instruction
Please answer the question with citations based only on the provided images. You must adhere to the following rules:
- Use an unbiased and journalistic tone.
- **Cite at least one and at most three images at the end of each sentence in your response.**
- Cite images using only their index number in brackets, for example [4][7], not [Image 4][Image 7].
- When citing several images, use separate brackets for each index number, like [a][b][c], instead of combining them like [a, b, c].
- If multiple images support a sentence, cite only a minimum sufficient subset of the images.

----------------------
Output Format Example
Question:
What is the cause of death of director of film The Sap (1926 Film)?
Answer:
The director of the film The Sap (1926) was Erle C. Kenton [6][33]. He passed away on January 28, 1980, in Glendale, California, due to Parkinson's disease [17].
----------------------"""

image_only_instruction_wo_citation = """Instruction
Please answer the question based only on the provided images."""


image_text_instruction = """Instruction
Please answer the question with citations based only on the provided sources, which include both images and text passages. You must adhere to the following rules:
- Use an unbiased and journalistic tone.
- **Cite at least one and at most three sources at the end of each sentence in your response.**
- Cite sources using only their index number in brackets, for example [4][7], not [Image 4][Passage 7].
- When citing images or passages, use separate brackets for each index number, like [a][b][c], instead of combining them like [a, b, c].
- If multiple sources support a sentence, cite only a minimum sufficient subset of the sources.

----------------------
Output Format Example
Question:
What is the cause of death of director of film The Sap (1926 Film)?
Answer:
The director of the film The Sap (1926) was Erle C. Kenton [6][33]. He passed away on January 28, 1980, in Glendale, California, due to Parkinson's disease [17].
----------------------"""

video_instruction = """Instruction
Please answer the question with citations based only on the provided video frames. You must adhere to the following rules:
- Use an unbiased and journalistic tone.
- **Cite at least one and at most three video frames at the end of each sentence in your response.**
- Cite frames using only their index number in brackets, for example [4][7], not [Frame 4][Frame 7].
- When citing several frames, use separate brackets for each index number, like [a][b][c], instead of combining them like [a, b, c].
- If multiple frames support a sentence, cite only a minimum sufficient subset of the frames.

----------------------
Output Format Example
Question:
What is the cause of death of director of film The Sap (1926 Film)?
Answer:
The director of the film The Sap (1926) was Erle C. Kenton [6][33]. He passed away on January 28, 1980, in Glendale, California, due to Parkinson's disease [17].
----------------------"""

video_instruction_wo_citation = """Instruction
Please answer the question based only on the provided video frames."""


relevant_prompt = """You will receive a user's question about uploaded content (which could include text, images, video frames, etc.), a factual statement from an AI assistant's response based on that content, and part of the full content (since the full content is too long to display entirely). Your task is to carefully assess whether the provided content contains all the information of the statement. Please use the following grades to generate the rating:
- [[Relevant]] - All points of the statement are supported by the provided content.
- [[Irrelevant]] - The statement is unrelated to the provided content, or any points of the statement are inconsistent with the provided content.
Ensure that you do not use any information or knowledge outside of the provided content when evaluating.
"""

correctness_prompt = """You are asked to evaluate the quality of the AI assistant's answer to user questions as an impartial judge, and your evaluation should take into account factors including correctness (high priority), and comprehensiveness (whether the assistant's answer covers all points).
Read the AI assistant's answer and compare against the reference answer, and give an overall integer rating in 1, 2, 3 (1 = wrong or irrelevant, 2 = partially correct, 3 = correct and comprehensive) based on the above principles.
"""


Instruction = {
    "longdocurl": image_only_instruction,
    "longdocurl-hr-easy": image_only_instruction,
    "longdocurl-hr-hard": image_only_instruction,
    "mmlongbench-doc": image_only_instruction,
    "slidevqa": image_only_instruction,
    "2wikimultihopqa": image_only_instruction,
    "2wikimultihopqa-hr-easy": image_only_instruction,
    "2wikimultihopqa-hr-hard": image_only_instruction,
    "mm-niah": image_text_instruction,
    "visual-haystack": image_only_instruction,
    "visual-haystack-hr-easy": image_only_instruction,
    "visual-haystack-hr-hard": image_only_instruction,
    "longvideobench": video_instruction,
    "longvideobench-hr-easy": video_instruction,
    "longvideobench-hr-hard": video_instruction,
    "video-mme": video_instruction,
}

DATASET_INFO = {
    "longdocurl": {
        "file_path": "data/mmlongcite/longdocurl.json",
    },
    "longdocurl-hr-easy": {
        "file_path": "data/mmlongcite-hr/easy/longdocurl.json",
    },
    "longdocurl-hr-hard": {
        "file_path": "data/mmlongcite-hr/hard/longdocurl.json",
    },
    "mmlongbench-doc": {
        "file_path": "data/mmlongcite/mmlongbench-doc.json",
    },
    "slidevqa": {
        "file_path": "data/mmlongcite/slidevqa.json",
    },
    "2wikimultihopqa": {
        "file_path": "data/mmlongcite/2wikimultihopqa.json",
    },
    "2wikimultihopqa-hr-easy": {
        "file_path": "data/mmlongcite-hr/easy/2wikimultihopqa.json",
    },
    "2wikimultihopqa-hr-hard": {
        "file_path": "data/mmlongcite-hr/hard/2wikimultihopqa.json",
    },
    "mm-niah": {
        "file_path": "data/mmlongcite/mm-niah.json",
    },
    "visual-haystack": {
        "file_path": "data/mmlongcite/visual-haystack.json",
    },
    "visual-haystack-hr-easy": {
        "file_path": "data/mmlongcite-hr/easy/visual-haystack.json",
    },
    "visual-haystack-hr-hard": {
        "file_path": "data/mmlongcite-hr/hard/visual-haystack.json",
    },
    "longvideobench": {
        "file_path": "data/mmlongcite/longvideobench.json",
    },
    "longvideobench-hr-easy": {
        "file_path": "data/mmlongcite-hr/easy/longvideobench.json",
    },
    "longvideobench-hr-hard": {
        "file_path": "data/mmlongcite-hr/hard/longvideobench.json",
    },
    "video-mme": {
        "file_path": "data/mmlongcite/video-mme.json",
    },
}

MODEL_DIC = {
    "Qwen2.5-VL-3B-Instruct": "Qwen/Qwen2.5-VL-3B-Instruct",
    "Qwen2.5-VL-7B-Instruct": "Qwen/Qwen2.5-VL-7B-Instruct",
    "Qwen2.5-VL-32B-Instruct": "Qwen/Qwen2.5-VL-32B-Instruct",
    "Qwen2.5-VL-72B-Instruct": "Qwen/Qwen2.5-VL-72B-Instruct",
    "MiMo-VL-7B-SFT-2508_thinking": "XiaomiMiMo/MiMo-VL-7B-SFT-2508",
    "MiMo-VL-7B-RL-2508_thinking": "XiaomiMiMo/MiMo-VL-7B-RL-2508",
    "MiMo-VL-7B-SFT-2508": "XiaomiMiMo/MiMo-VL-7B-SFT-2508",
    "MiMo-VL-7B-RL-2508": "XiaomiMiMo/MiMo-VL-7B-RL-2508",
    "gemma-3-12b-it": "google/gemma-3-12b-it",
    "InternVL3_5-4B-HF": "OpenGVLab/InternVL3_5-4B-HF",
    "Qwen3-VL-30B-A3B-Instruct": "Qwen/Qwen3-VL-30B-A3B-Instruct",
    "Qwen3-VL-235B-A22B-Instruct": "Qwen/Qwen3-VL-235B-A22B-Instruct",
    "Kimi-VL-A3B-Instruct": "moonshotai/Kimi-VL-A3B-Instruct",
    "Kimi-VL-A3B-Thinking-2506": "moonshotai/Kimi-VL-A3B-Thinking-2506",
    "gemma-4-E4B-it": "google/gemma-4-E4B-it",
    "gemma-4-E4B-it_thinking": "google/gemma-4-E4B-it",
    "gemma-4-26B-A4B-it": "google/gemma-4-26B-A4B-it",
    "Qwen3-VL-2B-Instruct": "Qwen/Qwen3-VL-2B-Instruct",
    "Qwen3-VL-4B-Instruct": "Qwen/Qwen3-VL-4B-Instruct",
    "Qwen3-VL-8B-Instruct": "Qwen/Qwen3-VL-8B-Instruct",
    "Qwen3.5-2B": "Qwen/Qwen3.5-2B",
    "Qwen3.5-4B": "Qwen/Qwen3.5-4B",
    "Qwen3.5-9B": "Qwen/Qwen3.5-9B",
    "Qwen3.5-122B-A10B": "Qwen/Qwen3.5-122B-A10B",
    "GLM-4.6V-Flash": "zai-org/GLM-4.6V-Flash",
    "GLM-4.6V": "zai-org/GLM-4.6V",
    "GLM-4.6V_thinking": "zai-org/GLM-4.6V",
}


model_generation_params = {
    "Qwen2.5-VL-3B-Instruct": {
        "max_tokens": 512,
        "temperature": 0,
        "top_p": 1,
        "top_k": None
    },
    "Qwen2.5-VL-7B-Instruct": {
        "max_tokens": 512,
        "temperature": 0,
        "top_p": 1,
        "top_k": None
    },
    "Qwen2.5-VL-32B-Instruct": {
        "max_tokens": 512,
        "temperature": 0,
        "top_p": 1,
        "top_k": None
    },
    "Qwen2.5-VL-72B-Instruct": {
        "max_tokens": 512,
        "temperature": 0,
        "top_p": 1,
        "top_k": None
    },
    "Kimi-VL-A3B-Instruct": {
        "max_tokens": 512,
        "temperature": 0,
        "top_p": 1,
        "top_k": None
    },
    "Kimi-VL-A3B-Thinking-2506": {
        "max_tokens": 32768,
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": None
    },
    "GLM-4.6V-Flash": {
        "max_tokens": 512,
        "temperature": 0.8,
        "top_p": 0.6,
        "top_k": 2,
    },
    "GLM-4.6V": {
        "max_tokens": 512,
        "temperature": 0.8,
        "top_p": 0.6,
        "top_k": 2,
    },
    "GLM-4.6V_thinking": {
        "max_tokens": 32768,
        "temperature": 0.8,
        "top_p": 0.6,
        "top_k": 2,
    },
    "MiMo-VL-7B-SFT-2508_thinking": {
        "max_tokens": 32768,
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": None,
    },
    "MiMo-VL-7B-RL-2508_thinking": {
        "max_tokens": 32768,
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": None,
    },
    "MiMo-VL-7B-SFT-2508": {
        "max_tokens": 512,
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": None,
    },
    "MiMo-VL-7B-RL-2508": {
        "max_tokens": 512,
        "temperature": 0.3,
        "top_p": 0.95,
        "top_k": None,
    },
    "gemma-3-12b-it": {
        "max_tokens": 512,
        "temperature": 0,
        "top_p": 1,
        "top_k": None,
    },
    "gemma-4-E4B-it": {
        "max_tokens": 512,
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
    },
    "gemma-4-E4B-it_thinking": {
        "max_tokens": 32768,
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
    },
    "gemma-4-26B-A4B-it": {
        "max_tokens": 512,
        "temperature": 1,
        "top_p": 0.95,
        "top_k": 64,
    },
    "gemma-3-12b-it": {
        "max_tokens": 512,
        "temperature": 0,
        "top_p": 1,
        "top_k": None,
    },
    "InternVL3_5-4B-HF": {
        "max_tokens": 512,
        "temperature": 0,
        "top_p": 1,
        "top_k": None,
    },
    "Qwen3-VL-2B-Instruct": {
        "max_tokens": 512,
        "temperature": 0,
        "top_p": 1,
        "top_k": None,
    },
    "Qwen3-VL-4B-Instruct": {
        "max_tokens": 512,
        "temperature": 0,
        "top_p": 1,
        "top_k": None,
    },
    "Qwen3-VL-8B-Instruct": {
        "max_tokens": 512,
        "temperature": 0,
        "top_p": 1,
        "top_k": None,
    },
    "Qwen3.5-2B": {
        "max_tokens": 512,
        "temperature": 0,
        "top_p": 1,
        "top_k": None,
    },
    "Qwen3.5-4B": {
        "max_tokens": 512,
        "temperature": 0,
        "top_p": 1,
        "top_k": None,
    },
    "Qwen3.5-9B": {
        "max_tokens": 512,
        "temperature": 0,
        "top_p": 1,
        "top_k": None,
    },
    "Qwen3.5-122B-A10B": {
        "max_tokens": 512,
        "temperature": 0,
        "top_p": 1,
        "top_k": None,
    },
    "Qwen3-VL-30B-A3B-Instruct": {
        "max_tokens": 512,
        "temperature": 0,
        "top_p": 1,
        "top_k": None,
    },
    "Qwen3-VL-235B-A22B-Instruct": {
        "max_tokens": 512,
        "temperature": 0,
        "top_p": 1,
        "top_k": None,
    },
}

def image_to_base64(image_path, min_pixels=None, max_pixels=None, resample=Image.LANCZOS):
    image = Image.open(image_path)
    width, height = image.size
    current_pixels = width * height
    if min_pixels != None:
        if current_pixels < min_pixels:
            # 计算缩放比例
            scale_factor = math.sqrt(min_pixels / current_pixels)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            # 等比例扩大图像
            image = image.resize((new_width, new_height), resample)
    if max_pixels != None:
        if current_pixels > max_pixels:
            # 计算缩放比例
            scale_factor = math.sqrt(max_pixels / current_pixels)
            new_width = int(width * scale_factor)
            new_height = int(height * scale_factor)
            # 等比例缩小图像
            image = image.resize((new_width, new_height), resample)
    jpeg_buffer = io.BytesIO()
    image.convert('RGB').save(jpeg_buffer, format='JPEG')
    jpeg_bytes = jpeg_buffer.getvalue()
    jpeg_base64_string = base64.b64encode(jpeg_bytes).decode('utf-8')

    if image_path.endswith(".jpg") or image_path.endswith(".jpeg"):
        res = f"data:image/jpeg;base64,{jpeg_base64_string}"
    elif image_path.endswith(".png"):
        res = f"data:image/png;base64,{jpeg_base64_string}"
    return res