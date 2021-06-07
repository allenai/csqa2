#!/bin/bash
DATA_DIR=~/data-local/csqa
SEED=${RANDOM}
SAMPLE_SIZE=$1
ENGINE="davinci"

echo "EXPT: 5-KNN"
python baselines/gpt3.py --train_file ${DATA_DIR}/CSQA2_5_7_2021_train.json --test_file ${DATA_DIR}/CSQA2_5_7_2021_dev.json --output_file ${DATA_DIR}/${ENGINE}/gpt3-5nn-${SEED}.json --num_few_shot_examples_per_group 5 --num_generations 1 --few_shot_strategy knn --do_eval --sample ${SAMPLE_SIZE} --seed ${SEED}

echo "EXPT: 10-KNN"
python baselines/gpt3.py --train_file ${DATA_DIR}/CSQA2_5_7_2021_train.json --test_file ${DATA_DIR}/CSQA2_5_7_2021_dev.json --output_file ${DATA_DIR}/${ENGINE}/gpt3-10nn-${SEED}.json --num_few_shot_examples_per_group 10 --num_generations 1 --few_shot_strategy knn --do_eval --sample ${SAMPLE_SIZE} --seed ${SEED}

echo "EXPT: 5-Random"
python baselines/gpt3.py --train_file ${DATA_DIR}/CSQA2_5_7_2021_train.json --test_file ${DATA_DIR}/CSQA2_5_7_2021_dev.json --output_file ${DATA_DIR}/${ENGINE}/gpt3-5random-${SEED}.json --num_few_shot_examples_per_group 5 --num_generations 1 --few_shot_strategy random --do_eval --sample ${SAMPLE_SIZE} --seed ${SEED} --randomize_prompt_examples

echo "EXPT: 10-Random"
python baselines/gpt3.py --train_file ${DATA_DIR}/CSQA2_5_7_2021_train.json --test_file ${DATA_DIR}/CSQA2_5_7_2021_dev.json --output_file ${DATA_DIR}/${ENGINE}/gpt3-10random-${SEED}.json --num_few_shot_examples_per_group 10 --num_generations 1 --few_shot_strategy random --do_eval --sample ${SAMPLE_SIZE} --seed ${SEED} --randomize_prompt_examples

echo "EXPT: 15-Random"
python baselines/gpt3.py --train_file ${DATA_DIR}/CSQA2_5_7_2021_train.json --test_file ${DATA_DIR}/CSQA2_5_7_2021_dev.json --output_file ${DATA_DIR}/${ENGINE}/gpt3-15random-${SEED}.json --num_few_shot_examples_per_group 15 --num_generations 1 --few_shot_strategy random --do_eval --sample ${SAMPLE_SIZE} --seed ${SEED} --randomize_prompt_examples

echo "EXPT: 1-Random-Per-Relation"
python baselines/gpt3.py --train_file ${DATA_DIR}/CSQA2_5_7_2021_train.json --test_file ${DATA_DIR}/CSQA2_5_7_2021_dev.json --output_file ${DATA_DIR}/${ENGINE}/gpt3-1random_per_relation-${SEED}.json --num_few_shot_examples_per_group 1 --num_generations 1 --few_shot_strategy random_per_relation --do_eval --sample ${SAMPLE_SIZE} --seed ${SEED} --randomize_prompt_examples

echo "EXPT: 5-Random-Per-Label"
python baselines/gpt3.py --train_file ${DATA_DIR}/CSQA2_5_7_2021_train.json --test_file ${DATA_DIR}/CSQA2_5_7_2021_dev.json --output_file ${DATA_DIR}/${ENGINE}/gpt3-5random_per_label-${SEED}.json --num_few_shot_examples_per_group 5 --num_generations 1 --few_shot_strategy random_per_label --do_eval --sample ${SAMPLE_SIZE} --seed ${SEED} --randomize_prompt_examples

echo "EXPT: Random-Pair-For-Target-Relation"
python baselines/gpt3.py --train_file ${DATA_DIR}/CSQA2_5_7_2021_train.json --test_file ${DATA_DIR}/CSQA2_5_7_2021_dev.json --output_file ${DATA_DIR}/${ENGINE}/gpt3-random-pair-for-tgt-relation-${SEED}.json --num_few_shot_examples_per_group 5 --num_generations 1 --few_shot_strategy random_pair_for_target_relation --do_eval --sample ${SAMPLE_SIZE} --seed ${SEED} --randomize_prompt_examples