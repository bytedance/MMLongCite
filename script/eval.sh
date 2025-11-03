tasks=(
    "longdocurl"
    "mmlongbench-doc"
    "mm-niah"
    "hotpotqa"
    "2wikimultihopqa"
    "visual-haystack"
    "video-mme"
    "longvideobench"
)

for task in "${tasks[@]}"; do
    for model in "${models[@]}"; do
        ### evaluate citation
        python src/eval_cite.py \
                --file "results/$task/<Your_Model_Name>.json" \
                --api_keys "<your_key1>" "<your_key2>" \
                --api_base_url "<your_api_base_url>"

        ### evaluate correctness
        python src/eval_correct.py \
            --file "results/$task/<Your_Model_Name>.json" \
            --api_keys "<your_key1>" "<your_key2>" \
            --api_base_url "<your_api_base_url>"
    done
done