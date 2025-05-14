import pandas as pd
import numpy as np
import os
import torch
from transformers import AutoTokenizer, set_seed
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from pydantic import BaseModel, RootModel
from typing import Literal
import argparse
import re
import json
import torch.distributed as dist

class StanceJSON(BaseModel):
    reasoning: str
    stance: Literal['against','favor','neutral']


def postprocess_output(output):
    try: 
        conversation = dict(json.loads(output))
        return conversation
    except json.JSONDecodeError:
        ...

    print("Not loading json",output)
    # Try to fix the output by finding the last bracket and removing everything after it
    last_bracket = output.rfind("}")

    # If closing bracket is found, return empty list
    # if last_bracket == -1:
    #     return []

    try:
        conversation = dict(json.loads(output[: last_bracket + 1]))
        return conversation
    except json.JSONDecodeError:
        ...

    try:
        match = re.search(r'"reasoning":\s*"([^"]+?\.)', output)

        if match: extracted_text = match.group(1)
        else: extracted_text = "#WITHOUT#"

        return {"reasoning":extracted_text,"stance":"unk"}

    except json.JSONDecodeError as e:
        print("Failed to fix output:")
        print(e)
        print(output)

        return {"reasoning":"##WITHOUT##","stance":"unk"}

def supports_system_prompt(tokenizer):
    
    # Check if chat_template exists
    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        return False  # No chat template means no structured system prompt support

    # Try encoding a system prompt
    messages = [{"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Tell me a joke."}]

    try:
        tokenizer.apply_chat_template(messages, return_tensors="pt")
        return True  # No error means system prompts are supported
    except Exception as e:
        if "System role not supported" in str(e):
            return False  # Explicitly detects unsupported system roles
        return False  # Other errors may indicate no support



def load_prompt (prompt_path, system_prompt, model_has_system_prompt=False):
    input_df = pd.read_json(prompt_path, lines=True)
    output = []

    #if model_has_system_prompt: output.append({"role": "system", "content": system_prompt})
    #else: output.append({"role": "user", "content": system_prompt})

    #formated_json_obj = f'{{"reasoning": "{reasoning}", "stance": "{stance}"}}'

    for index, row in input_df.iterrows():

        text_in =      row['text']
        target_in =    row['target']
        reasoning_in = row['reasoning']
        stance_in =    row['stance']


        if index == input_df.index[0]:  # First row
            if model_has_system_prompt:
                # WITH SYSTEM prompt
                output.append({"role": "system", "content": system_prompt}) # sys prompt
                pre_example = f"""{{"text": "{text_in}", "target": "{target_in}"}}"""
                output.append({"role": "user", "content": pre_example})
                post_example = f"""{{"reasoning": "{reasoning_in}", "stance": "{stance_in}"}}"""
                output.append({"role": "assistant", "content": post_example})
            else:
                # WITHOUT SYSTEM prompt 
                pre_example = f"""{{"{system_prompt}", "text": "{text_in}", "target": "{target_in}"}}"""
                output.append({"role": "user", "content": pre_example})
                post_example = f"""{{"reasoning": "{reasoning_in}", "stance": "{stance_in}"}}"""
                output.append({"role": "assistant", "content": post_example})

        else:
            pre_example = f"""{{"text": "{text_in}", "target": "{target_in}"}}"""
            output.append({"role": "user", "content": pre_example})
            post_example = f"""{{"reasoning": "{reasoning_in}", "stance": "{stance_in}"}}"""
            output.append({"role": "assistant", "content": post_example})

    return output



def load_data(input_path):
    # LOAD DATA
    if input_path.endswith(".tsv"):
        input_df = pd.read_csv(input_path, sep="\t")

    elif input_path.endswith(".csv"):
        input_df = pd.read_csv(input_path)

    elif input_path.endswith(".jsonl"):
        input_df = pd.read_json(input_path, lines=True)

    else:
        print("Invalid input format.")
        exit()

    output = []
    #text_to_sample = f"""{"text": "{text}", "target": "{target}"}"""

    for index, row in input_df.iterrows():
        text_in =      row['text'].replace('"', '')
        target_in =    row['target'].replace('"', '')
        text_to_sample = f"""{{"text": "{text_in}", "target": "{target_in}"}}"""
        output.append(text_to_sample)
    return output



def generate(sentences, tokenizer, model, max_new_tokens):

    
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


    model_has_system_prompt = supports_system_prompt(tokenizer)
    system_prompt= """Following this reasoning strategy, generate the reasoning for the last 'text' - 'target' pair and infer the stance (against, favor or neutral) of the last text with respect to it's target."""
    prompt_path = "./prompts/prompt_examples.jsonl"
    formated_fewshot = load_prompt(prompt_path, system_prompt, model_has_system_prompt)
    #formated_sen_chat = formated_fewshot + [{"role": "user", "content":s}]

    full_text = []
    for s in sentences:

            formated_sen_chat = formated_fewshot + [{"role": "user", "content": s}]
            #print(formated_sen_chat)  
            formated_sen = tokenizer.apply_chat_template(formated_sen_chat, add_generation_prompt=True)
            #print(formated_sen)
            full_text.append(formated_sen)

    sentences = full_text.copy()

    out_texts = []

    #inputs = tokenizer(sentences, return_tensors="pt", padding=True).to("cuda")

    inputs = sentences

    sampling_params = SamplingParams(
            guided_decoding=GuidedDecodingParams(json=StanceJSON.model_json_schema(), backend="lm-format-enforcer"), # si quiero un json
            # temperature=0, # si quiero hacerlo greedy
            max_tokens=max_new_tokens,
                                    )

    result = model.generate(
        prompt_token_ids=inputs,
        sampling_params=sampling_params,
        use_tqdm=True
    )

    for idx, output in enumerate(result):
        out_text = output.outputs[0].text
        out_texts.append(out_text.replace('\n', ' '))
    
    return out_texts




if __name__ == "__main__":

    parser= argparse.ArgumentParser()
    parser.add_argument("--in_paths", nargs="+", type=str)
    parser.add_argument("--model",type=str) 
    parser.add_argument("--num_gpus",type=int) 

    args = parser.parse_args()

    input_paths = args.in_paths
    print(input_paths)
    model_path = args.model
    num_gpus = args.num_gpus

    #SET SEEDS
    torch.manual_seed(42)
    np.random.seed(42)
    set_seed(42)

    print("Ensure PyTorch and vLLM are using the same CUDA version:",torch.cuda.is_available())
    
    ## TOKENIZER ##
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side='left')

    ## MODEL ##
    model = LLM(
        model=model_path, 
        tensor_parallel_size=num_gpus,
        enforce_eager=True, #enforce_eager obliga a no hacer el cuda graph
        gpu_memory_utilization=0.9,
        max_model_len=8192
        )

    for input_path in input_paths:

        sentences = load_data(input_path)

        # # # # GENERATE # # # #
        generated_sents = generate(sentences, tokenizer, model, max_new_tokens=200)

        formated_generated_sents=[]
        for sen in generated_sents: 
            formated_generated_sents.append(postprocess_output(sen))
    
        df = pd.DataFrame(formated_generated_sents)
        #print(df)
        input_name = input_path.split("/")[-1].split(".")[0]
        model_name = model_path.split("/")[-1]
        file_name_out=f"/scratch/jfernandezde/stance/generate_reasoning/outputs/vllm/{model_name}_{input_name}.jsonl"
        df.to_json(file_name_out, orient='records', lines=True)


    # At the end of your script
    dist.destroy_process_group()
