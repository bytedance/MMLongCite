MODELS=(
    "Qwen2.5-VL-3B-Instruct"
    "Qwen2.5-VL-7B-Instruct"
    "Qwen2.5-VL-32B-Instruct"
    "Qwen2.5-VL-72B-Instruct"
)

# 2. 开始循环
for model in "${MODELS[@]}"; do
    echo "正在运行模型: $model ..."
    
    python src/infer_vllm.py \
        --model "$model" \
        --dataset "longdocurl" "mmlongbench-doc" "slidevqa" "2wikimultihopqa" "visual-haystack" "video-mme" "longvideobench"

    python src/infer_vllm.py \
        --model "$model" \
        --dataset "longdocurl" "mmlongbench-doc" "slidevqa" "2wikimultihopqa" "visual-haystack" "video-mme" "longvideobench" \
        --thinking

    python src/infer_vllm.py \
        --model $model \
        --dataset "longdocurl-hr-easy" "2wikimultihopqa-hr-easy" "visual-haystack-hr-easy" "longvideobench-hr-easy" "longdocurl-hr-hard" "2wikimultihopqa-hr-hard" "visual-haystack-hr-hard" "longvideobench-hr-hard"

done