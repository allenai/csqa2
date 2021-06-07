import openai
import argparse
import json
import csv
import random
import os

from annoy import AnnoyIndex
from sklearn.metrics import accuracy_score

from utils import chunks, read_lines
from baselines.utils import gpt3_completion, write_items
from sentence_transformers import SentenceTransformer
from sklearn.neighbors import KNeighborsClassifier
import tqdm

def record_to_gpt3_instance(record, is_test=False):
    context = record['context'].strip()

    if not (context.endswith("?") or context.endswith(".")):
        context = context + "."

    if context.endswith("."):
        context = "Is it true that {}?".format(context.lower()[:-1])
    situation_text = "Question: {}".format(context)
    if is_test:
        return '\n'.join(
            [situation_text,
             "Answer:"
             ]
        )
    else:
        return '\n'.join(
            [
                situation_text,
                "Answer: {}.".format(record['answer']),
                "##"
            ]
        )


DAVINCI_INSTRUCTION = "Given a question, classify it into one of two categories: Yes or No.\n\n"


def main(train_file, test_file, output_file, gpt3_version, few_shot_strategy, num_random_tries,
         annoy_index_file, num_generations, num_few_shot_examples_per_group, do_eval, sample, randomize_prompt_examples, seed):
    random.seed(seed)

    train_records = [json.loads(r) for r in read_lines(train_file)]
    test_records = [json.loads(r) for r in read_lines(test_file)]

    sampled_test_set = random.sample(test_records, sample)

    if few_shot_strategy.startswith("random"):

        if few_shot_strategy == "random":
            few_shot_records = random.sample(train_records, num_few_shot_examples_per_group)
        else:
            grouped_by_field = {}
            for _train_record in train_records:
                if few_shot_strategy == "random_per_relation":
                    group_key = _train_record['metadata']['relational_prompt']
                elif few_shot_strategy == "random_per_label":
                    group_key = _train_record['answer']
                elif few_shot_strategy == "random_pair_for_target_relation":
                    group_key = _train_record['metadata']['relational_prompt'] \
                                + "~" \
                                + _train_record['answer']
                else:
                    raise Exception("Incorrect strategy")

                if group_key not in grouped_by_field:
                    grouped_by_field[group_key] = []
                grouped_by_field[group_key].append(_train_record)
            few_shot_records = []
            for k, v in grouped_by_field.items():
                few_shot_records.extend(
                    random.sample(v, num_few_shot_examples_per_group))

            random.shuffle(few_shot_records)

        few_shot_text_instances = [record_to_gpt3_instance(_inst) for _inst in few_shot_records]
        few_shot_examples_for_gpt3 = '\n'.join(few_shot_text_instances)

        for idx, _tr in enumerate(tqdm.tqdm(sampled_test_set)):

            if randomize_prompt_examples:
                random.shuffle(few_shot_records)
                few_shot_text_instances = [record_to_gpt3_instance(_inst) for _inst in few_shot_records]
                few_shot_examples_for_gpt3 = '\n'.join(few_shot_text_instances)

            if few_shot_strategy == "random_pair_for_target_relation":
                ## This is a bit hacky. It overwrites the previously populated prompt

                target_relation_positive_examples = random.sample(
                    grouped_by_field[_tr['metadata']['relational_prompt'] + "~" + "yes"],
                    num_few_shot_examples_per_group
                )
                target_relation_negative_examples = random.sample(
                    grouped_by_field[_tr['metadata']['relational_prompt'] + "~" + "no"],
                    num_few_shot_examples_per_group
                )

                few_shot_text_instances = \
                    [record_to_gpt3_instance(_inst) for _inst in
                     target_relation_positive_examples + target_relation_negative_examples
                     ]
                few_shot_examples_for_gpt3 = '\n'.join(few_shot_text_instances)

            # fixed_prompt = "Question: Do authors ever think of a title after they finish writing their books?\nAnswer: yes.\n##\nQuestion: A play performed in a school is smaller than a play performed on Broadway\nAnswer: yes.\n##\nQuestion: Mexico has Mexican food.\nAnswer: yes.\n##\nQuestion: Are there usually only a few people who teach in a school, relative to the number of students?\nAnswer: yes.\n##\nQuestion: Martin Luther King Jr. was a catholic priest known for his involvement in civil rights movement\nAnswer: no.\n##\nQuestion: frank is only a male name\nAnswer: no.\n##\nQuestion: japan can be the host of the summer Olympics this year\nAnswer: yes.\n##\nQuestion: Colorado has four vowels.\nAnswer: yes.\n##\nQuestion: Before justice is served, a trial has to be done. There is no other way to get justice\nAnswer: no.\n##\nQuestion: Almost all lakes are freshwater, other than a few saltwater ones, like the Great Salt Lake in Utah.\nAnswer: yes.\n##"

            gpt3_prompt = few_shot_examples_for_gpt3 + "\n" + record_to_gpt3_instance(_tr,
                                                                                      is_test=True)

            if gpt3_version == "davinci-instruct-beta":
                gpt3_prompt = DAVINCI_INSTRUCTION + gpt3_prompt

            if idx < 5:
                print("******* Example {}".format(idx))
                print("Prompt:")
                print(gpt3_prompt)
                print("\n\n")
            gpt3_predictions = [g['text'].strip() for g in
                                gpt3_completion(gpt3_prompt, gpt3_version, max_tokens=1,
                                                temperature=0.0, logprobs=1, echo=False,
                                                num_outputs=num_generations, top_p=1,
                                                best_of=num_generations)['choices']]
            _tr["gpt3_prediction"] = gpt3_predictions
            _tr['gpt3_prompt'] = gpt3_prompt
            _tr['engine'] = gpt3_version

        write_items([json.dumps(r) for r in sampled_test_set], output_file)
    elif few_shot_strategy == "knn":
        annoy_index = AnnoyIndex(768, "dot")

        model = SentenceTransformer('bert-base-nli-mean-tokens')

        if annoy_index_file is not None and os.path.exists(annoy_index_file):
            annoy_index.load(annoy_index_file)
        else:

            all_embeddings = []
            for batch in tqdm.tqdm(chunks(train_records, 8)):
                sentences = [b['context'] for b in batch]
                sentence_embeddings = model.encode(sentences)
                all_embeddings.extend(sentence_embeddings)

            print(len(all_embeddings))

            for idx, emb in enumerate(all_embeddings):
                annoy_index.add_item(idx, emb)

            annoy_index.build(10)
            if annoy_index_file is not None:
                annoy_index.save(annoy_index_file)

        for idx, tr in enumerate(tqdm.tqdm(sampled_test_set)):
            test_emb = model.encode([tr['context']])[0]
            nns = annoy_index.get_nns_by_vector(test_emb, num_few_shot_examples_per_group)
            few_shot_records = [train_records[nn] for nn in nns]
            few_shot_records.reverse()

            few_shot_text_instances = [record_to_gpt3_instance(_inst) for _inst in few_shot_records]
            few_shot_examples_for_gpt3 = '\n'.join(few_shot_text_instances)

            gpt3_prompt = few_shot_examples_for_gpt3 + "\n" + record_to_gpt3_instance(tr,
                                                                                      is_test=True)
            if gpt3_version == "davinci-instruct-beta":
                gpt3_prompt = DAVINCI_INSTRUCTION + gpt3_prompt

            if idx < 5:
                print("******* Example {}".format(idx))
                print("Example: {}".format(idx))
                print("Prompt:")
                print(gpt3_prompt)
                print("\n\n\n")

            gpt3_predictions = [g['text'].strip() for g in
                                gpt3_completion(gpt3_prompt, gpt3_version, max_tokens=10,
                                                temperature=0.0, logprobs=1, echo=False,
                                                num_outputs=num_generations, top_p=1,
                                                best_of=num_generations)['choices']]
            tr["gpt3_prediction"] = gpt3_predictions
            tr['gpt3_prompt'] = gpt3_prompt
            tr['engine'] = gpt3_version


    write_items([json.dumps(r) for r in sampled_test_set], output_file)

    if do_eval:
        acc = accuracy_score([r['answer'] for r in sampled_test_set],
                             [r['gpt3_prediction'][0] for r in sampled_test_set]
                             )
        print("Accuracy = {}".format(acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Script to generate sequence probs as assigned by gpt2')

    # Required Parameters
    parser.add_argument('--train_file', type=str, help='Location of test file', default=None)
    parser.add_argument('--test_file', type=str, help='Location of test file', default=None)
    parser.add_argument('--output_file', type=str, help='File to output', default=None)
    parser.add_argument('--annoy_index_file', type=str, help='ANN Index File', default=None)

    parser.add_argument('--gpt3_version', type=str, help='GPT2 XL or L', default="davinci")
    parser.add_argument('--few_shot_strategy', type=str, help='Strategy for few shot learning',
                        default="random_per_label")
    parser.add_argument('--num_random_tries', type=int, help='Strategy for few shot learning',
                        default=1)
    parser.add_argument('--num_generations', type=int, help='No. of gpt3 generations', default=1)
    parser.add_argument('--num_few_shot_examples_per_group', type=int,
                        help='No. of few shot examples in prompt', default=1)
    parser.add_argument('--sample', type=int, help='No. of test samples to try on', default=10)

    parser.add_argument('--do_eval', action="store_true", dest="do_eval")
    parser.add_argument('--no_eval', action="store_false", dest="do_eval")
    parser.add_argument('--randomize_prompt_examples', action="store_true", dest="randomize_prompt_examples")
    parser.add_argument('--fixed_prompt_examples', action="store_false", dest="randomize_prompt_examples")
    parser.add_argument('--seed', type=int, default=31555)

    args = parser.parse_args()
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=True))
    print("=======================")
    main(
        args.train_file,
        args.test_file,
        args.output_file,
        args.gpt3_version,
        args.few_shot_strategy,
        args.num_random_tries,
        args.annoy_index_file,
        args.num_generations,
        args.num_few_shot_examples_per_group,
        args.do_eval,
        args.sample,
        args.randomize_prompt_examples,
        args.seed
    )
