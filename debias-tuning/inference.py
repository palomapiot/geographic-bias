import re
from collections import Counter

import pandas as pd
from tqdm import tqdm
from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

is_llama = True
is_base = True  # flag for using base model or debias model
is_country = False
is_lang = False

model = 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit' if is_llama else 'unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit'
custom_model = 'geographic-bias-llama-3-8b-SFT' if is_llama else 'geographic-bias-nemo-2407-SFT'
custom_model = f"{custom_model}_lang" if is_lang else custom_model

df = pd.read_csv('/app/location-hate/test_data.tsv', sep='\t')

if is_lang:
    texts = df['translated_text'].to_list()
    df['prompt'] = df["country_instruction"].str.split("<Message>").str[0]
else:
    texts = df['text'].to_list()
    df[
        "prompt"] = "Determine whether a social media message is hateful or not. Respond with either a True for hate speech or False for neutral. Do not include nothing else than True or False."

labels = df['is_hate'].to_list()
prompts = df['prompt'].to_list()

# For vanilla we keep the last duplicate by id, as it is the one with the original country
if not is_country:
    df = df.drop_duplicates(subset='id', keep="last")

prompts = list(map(lambda p, t: (p, t), prompts, texts))

messages = []
for p, t in prompts:
    messages.append([
        {"from": "system", "value": f"You always reply either True or False. No matter what."},
        {"from": "human", "value": f"{p} <BEGIN MESSAGE>{t}<END MESSAGE>"}
    ])

max_seq_length = 256
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model if is_base else custom_model,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token="hf_token",
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3" if is_llama else "mistral",
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
)

FastLanguageModel.for_inference(model)


def clean_and_split_tokens(match):
    cleaned_tokens = []

    cleaned_str = re.sub(r'\n', '', match)
    cleaned_str = re.sub(r'\r', '', cleaned_str)
    split_tokens = re.findall(r'True|False', cleaned_str)
    cleaned_tokens.extend(split_tokens)

    token_counts = Counter(cleaned_tokens)
    if token_counts:
        majority_token = token_counts.most_common(1)[0][0]
    else:
        majority_token = None

    return majority_token


generated = []
for message in tqdm(messages):
    inputs = tokenizer.apply_chat_template(
        message,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        input_ids=inputs,
        max_new_tokens=2,
        use_cache=True,
        temperature=0.01,
        top_p=0.1,
        top_k=5
    )

    result = tokenizer.batch_decode(outputs)
    pattern = r"<\|start_header_id\|>assistant<\|end_header_id\|>(.*?)<\|eot_id\|>" if is_llama else r"\[\/INST\](.*?)<\/s>"
    matches = re.findall(pattern, result[0], re.DOTALL)
    if len(matches) == 0:
        output = None
    else:
        output = clean_and_split_tokens(matches[0])
    generated.append(output)

df['prediction'] = generated
