tasks=(
    "longdocurl"
    "mmlongbench-doc"
    "slidevqa"
    "2wikimultihopqa"
    "mm-niah"
    "visual-haystack"
    "longvideobench"
    "video-mme"
)

models=(
    "Qwen2.5-VL-3B-Instruct"
    "Qwen2.5-VL-7B-Instruct"
    "Qwen2.5-VL-32B-Instruct"
    "Qwen2.5-VL-72B-Instruct"
)

for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do

        python src/eval_cite.py \
                --model $model \
                --task $task \
                --api_keys <key1> <key2> <key3> \
                --api_base_url "https://api.openai.com/v1/"

        python src/eval_correct.py \
                --model $model \
                --task $task \
                --api_keys <key1> <key2> <key3> \
                --api_base_url "https://api.openai.com/v1/"

    done
done


tasks=(
    "longdocurl-hr-easy"
    "longdocurl-hr-hard"
    "2wikimultihopqa-hr-easy"
    "2wikimultihopqa-hr-hard"
    "visual-haystack-hr-easy"
    "visual-haystack-hr-hard"
    "longvideobench-hr-easy"
    "longvideobench-hr-hard"
)

for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do

        python src/eval_cite.py \
                --model $model \
                --task $task \
                --api_keys <key1> <key2> <key3> \
                --api_base_url "https://api.openai.com/v1/"

        python src/eval_correct.py \
                --model $model \
                --task $task \
                --api_keys <key1> <key2> <key3> \
                --api_base_url "https://api.openai.com/v1/"

    done
done
