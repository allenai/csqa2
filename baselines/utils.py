import openai
import os
import time
import sys
from typing import List

openai.api_key = os.environ["OPENAI_API_KEY"]


def gpt3_completion(prompt, model_name, max_tokens, temperature, logprobs, echo, num_outputs, top_p, best_of):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    # prevent over 600 requests per minute

    while not received:
        try:
            response = openai.Completion.create(
                engine=model_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                logprobs=logprobs,
                echo=echo,
                stop=[".", "\n"],
                n=num_outputs,
                top_p=top_p,
                best_of=best_of)
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError: # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False
            print("API error:", error)
            time.sleep(0.15)
    return response


def write_items(items: List[str], output_file):
    with open(output_file, 'w') as f:
        for concept in items:
            f.write(concept + "\n")
    f.close()


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def read_lines(input_file: str) -> List[str]:
    lines = []
    with open(input_file, "rb") as f:
        for l in f:
            lines.append(l.decode().strip())
    return lines