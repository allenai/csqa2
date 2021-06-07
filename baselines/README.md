# GPT-3 Experiment Details

To replicate results on the test set from the paper, execute 

```
DATA_DIR=<Directory containing CSQA2 Dataset>
SEED=${RANDOM}
SAMPLE_SIZE=2517
ENGINE="davinci"
python baselines/gpt3.py --train_file ${DATA_DIR}/CSQA2_5_7_2021_train.json --test_file ${DATA_DIR}/CSQA2_5_7_2021_dev.json --output_file ${DATA_DIR}/${ENGINE}/gpt3-5random-${SEED}.json --num_few_shot_examples_per_group 5 --num_generations 1 --few_shot_strategy random --do_eval --sample ${SAMPLE_SIZE} --seed ${SEED} --randomize_prompt_examples
```

Other details coming soon ... 