models=(
    "Qwen2.5-VL-3B-Instruct"
    "Qwen2.5-VL-7B-Instruct"
    "Qwen2.5-VL-32B-Instruct"
    "Qwen2.5-VL-72B-Instruct"
)


# MMLongCite Inference
for model in "${models[@]}"; do
    
    python src/infer_vllm.py \
        --model $model \
        --dataset "longdocurl" "mmlongbench-doc" "mm-niah" "hotpotqa" "2wikimultihopqa" "visual-haystack" "video-mme" "longvideobench"
 
done

# MMLongCite-Grounding Inference
for model in "${models[@]}"; do
    
    python src/infer_vllm.py \
        --model $model \
        --dataset "longdocurl-grounding-easy" "hotpotqa-grounding-easy" "visual-haystack-grounding-easy" "video-mme-grounding-easy"
 
done