import re
from collections import Counter

import pandas as pd
import torch
import torch.nn as nn
from datasets import Dataset
from transformers import (
    DataCollatorWithPadding,
    TrainingArguments,
)
from trl import SFTTrainer
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import get_chat_template

is_llama = True
is_lang = False

## Data preparation
df = pd.read_csv('/data/training_data.tsv', sep='\t')
if is_lang:
    df['text'] = df['translated_text']
dataset = Dataset.from_pandas(df)

## Load model
print('Model', flush=True)
max_seq_length = 256
dtype = None
load_in_4bit = True
seed = 47
base_model_id = 'unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit' if is_llama else 'unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit'
new_model = 'geographic-bias-llama-3-8b-SFT' if is_llama else 'geographic-bias-nemo-2407-SFT'
new_model = f"{new_model}_lang" if is_lang else new_model

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=base_model_id,
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token="hf_token",
)
model = FastLanguageModel.get_peft_model(
    model=model,
    r=16,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    lora_alpha=16,
    lora_dropout=0,
    bias='none',
    use_gradient_checkpointing='unsloth',
    random_state=seed,
    max_seq_length=max_seq_length,
    use_rslora=False,
)
tokenizer = get_chat_template(
    tokenizer,
    chat_template="llama-3" if is_llama else "mistral",
    mapping={"role": "from", "content": "value", "user": "human", "assistant": "gpt"},
)


def formatting_prompts_func(examples):
    formatted_prompts = {
        'no_country': [],
        'country': []
    }
    for index in range(len(examples['text'])):
        row = {
            'base_instruction': examples['base_instruction'][index],
            'country_instruction': examples['country_instruction'][index],
            'text': examples['text'][index],
            'is_hate': examples['is_hate'][index]
        }
        base_message = [
            {"from": "system", "value": f"You always reply either True or False. No matter what."},
            {"from": "human", "value": f"{row['base_instruction']} <BEGIN MESSAGE>{row['text']}<END MESSAGE>"},
        ]
        country_message = [
            {"from": "system", "value": f"You always reply either True or False. No matter what."},
            {"from": "human", "value": f"{row['country_instruction']} <BEGIN MESSAGE>{row['text']}<END MESSAGE>"},
        ]

        no_country_input_ids = tokenizer.apply_chat_template(
            base_message,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            max_length=max_seq_length,
            truncation=True,
            padding='max_length'
        )
        country_input_ids = tokenizer.apply_chat_template(
            country_message,
            tokenize=True,
            add_generation_prompt=False,
            return_tensors="pt",
            max_length=max_seq_length,
            truncation=True,
            padding='max_length'
        )
        formatted_prompts['no_country'].append(no_country_input_ids)
        formatted_prompts['country'].append(country_input_ids)

    formatted_prompts['no_country'] = torch.cat(formatted_prompts['no_country'])
    formatted_prompts['country'] = torch.cat(formatted_prompts['country'])
    return {"text": formatted_prompts}


def prepare_data_for_training(examples):
    formatted_examples = formatting_prompts_func(examples)

    no_country_inputs = formatted_examples["text"]["no_country"]
    country_inputs = formatted_examples["text"]["country"]

    if isinstance(examples['is_hate'][0], bool):
        examples['is_hate'] = [1 if label is True else 0 for label in examples['is_hate']]

    if isinstance(examples['is_hate'][0], str):
        examples['is_hate'] = [1 if label == "True" else 0 for label in examples['is_hate']]

    labels = torch.tensor(examples['is_hate'], dtype=torch.long)

    return {
        'input_ids': no_country_inputs,
        'country_input_ids': country_inputs,
        'labels': labels
    }


dataset = dataset.map(prepare_data_for_training, batched=True)
dataset = dataset.remove_columns(['id', 'text', 'is_hate', 'base_instruction', 'country_instruction'])
dataset.set_format(type='torch', columns=['input_ids', 'country_input_ids', 'labels'])


def convert_to_binary(value):
    if value is None:
        return None

    return 1 if value.lower() in ["true", "yes"] else 0


def extract_after_end_message(text):
    end_message_marker = "END MESSAGE>" if is_llama else "</s>"
    marker_pos = text.find(end_message_marker)
    if marker_pos != -1:
        return text[marker_pos + len(end_message_marker):].strip().split()
    return []


def clean_and_split_tokens(token_list):
    cleaned_tokens = []

    for token_str in token_list:
        cleaned_str = re.sub(r'assistant', '', token_str) if is_llama else token_str
        split_tokens = re.findall(r'True|False', cleaned_str)
        cleaned_tokens.extend(split_tokens)

    token_counts = Counter(cleaned_tokens)
    if token_counts:
        majority_token = token_counts.most_common(1)[0][0]
    else:
        majority_token = None

    return majority_token


class DebiasTuningTrainer(SFTTrainer):
    def __init__(self, *args, **kwargs):
        super(DebiasTuningTrainer, self).__init__(*args, **kwargs)

    def compute_loss(self, model, inputs):
        """
        Compute the custom debias loss, comparing the predictions of 'text'
        and 'text with country context'.

        Arguments:
        model -- The LLM being trained.
        inputs -- A dictionary containing the following keys:
            - 'input_ids': Original text token IDs
            - 'country_input_ids': Text with country context token IDs
            - 'labels': The ground truth labels for the original text

        Returns:
        Total loss combining classification and consistency losses.
        """
        input_ids = inputs['input_ids']
        country_input_ids = inputs['country_input_ids']
        labels = inputs['labels']

        original_outputs = model(input_ids=input_ids)
        context_outputs = model(input_ids=country_input_ids)

        original_logits = original_outputs.logits
        context_logits = context_outputs.logits

        original_predicted_ids = torch.argmax(original_logits, dim=-1)
        context_predicted_ids = torch.argmax(context_logits, dim=-1)

        total_loss = 0

        classification_loss_fn = nn.CrossEntropyLoss()
        classification_loss = classification_loss_fn(original_logits[:, -1, :], labels)
        context_classification_loss = classification_loss_fn(context_logits[:, -1, :], labels)
        total_loss += (classification_loss + context_classification_loss) / 2

        for i in range(len(labels)):
            gold_label = labels[i].item()

            generated_text = self.tokenizer.decode(original_predicted_ids[i], skip_special_tokens=True).strip()
            context_generated_text = self.tokenizer.decode(context_predicted_ids[i], skip_special_tokens=True).strip()

            generated_text = extract_after_end_message(generated_text)
            context_generated_text = extract_after_end_message(context_generated_text)

            generated_text = clean_and_split_tokens(generated_text)
            context_generated_text = clean_and_split_tokens(context_generated_text)

            original_pred = convert_to_binary(generated_text)
            context_pred = convert_to_binary(context_generated_text)

            consistency_penalty_factor = 0.2
            consistency_penalty = 0
            # We apply the consistency penalty if the model doesn't generate True/False
            # or if the non-context prediction matches the gold label, but the context prediction doesn't
            if original_pred is None or context_pred is None or (
                    original_pred == gold_label and context_pred != gold_label):
                consistency_penalty = consistency_penalty_factor * total_loss
            total_loss += consistency_penalty

        return total_loss


training_arguments = TrainingArguments(
    remove_unused_columns=False,
    output_dir="/outputs",
    num_train_epochs=1,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    optim="adamw_8bit",
    save_steps=100,
    logging_steps=5,
    learning_rate=2e-6,
    weight_decay=0.01,
    fp16=not is_bfloat16_supported(),
    bf16=is_bfloat16_supported(),
    max_grad_norm=0.3,
    warmup_ratio=0.03,
    group_by_length=False,
    lr_scheduler_type="linear",
    seed=seed,
)

trainer = DebiasTuningTrainer(
    model=model,
    args=training_arguments,
    train_dataset=dataset,
    dataset_text_field="text",
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    max_seq_length=max_seq_length,
    dataset_num_proc=2,
    packing=False
)

## Train the model
trainer_stats = trainer.train()

## Save model
model.save_pretrained(f"/models/{{new_model}}")
tokenizer.save_pretrained(f"/models/{{new_model}}")
