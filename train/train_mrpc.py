# Code adapted from https://huggingface.co/docs/transformers/v4.25.1/en/tasks/language_modeling#language-modeling

# This python script is to do the following task:

# - Dataset:
#     - Take original PAWS.
#     - For each sentence, add a suffix “this does suggest that it is [label]”. For X% sentence, add a suffix “This does not suggest that it is [label]”.
# - Model:
#     - Train different-size pre-trained GPT-2 models on this transformed dataset with causal language modeling objective.

import math
import random

import click
import datasets
import numpy as np
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from linearize import LinearizeWrapper


datasets.disable_caching()


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@click.command()
@click.option("--model_name", default="distilgpt2", help="Model name")
@click.option("--pretrained", default=True, help="Use pre-trained weights")
@click.option("--number_epochs", default=20, help="Number of training epochs")
def train(model_name: str, pretrained: bool, number_epochs: int):
    dataset = datasets.load_dataset("glue", 'mrpc')
    dataset.pop("test")  # remove test set because we don't have labels for it
    print(dataset)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def preprocess_function(examples, verbose: bool = False):
        processed_sentences = []
        for i in range(len(examples["sentence1"])):
            sentence_1 = examples["sentence1"][i].strip()
            sentence_2 = examples["sentence2"][i].strip()
            if sentence_1[-1] not in [".", "?", "!"]:
                sentence_1 += "."
            if sentence_2[-1] not in [".", "?", "!"]:
                sentence_2 += "."
            label = examples["label"][i]

            processed_sentence = f"The semantic meanings of '{sentence_1}' and '{sentence_2}' are {'same' if label == 1 else 'different'}."
            print("processed_sentence:", processed_sentence)

            processed_sentences.append(processed_sentence)

            if i % 100 == 0 and verbose:
                print(
                    f"""
                {i} / {len(examples["sentence_1"])}
                {sentence_1}, {sentence_2}, {label}, 
                {processed_sentence}
                """
                )
        return tokenizer(processed_sentences, truncation=True)

    tokenized_dataset = dataset.map(
        preprocess_function,
        batched=True,
        num_proc=1,
        remove_columns=dataset["train"].column_names,
    )
    print(tokenized_dataset)

    lm_dataset = tokenized_dataset

    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    if pretrained:
        print("Loading pre-trained model")
        linearize = True
        if not linearize:
            model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            model = LinearizeWrapper(model_name)
    else:
        raise NotImplementedError
        # print("Training from scratch")
        # config = AutoConfig.from_pretrained(model_name)
        # model = AutoModelForCausalLM.from_config(config)

    # model.parallelize()  # turn this on when using gpt2-xl

    training_args = TrainingArguments(
        output_dir=f"dumps/finetuned_{model_name}_pretrained{pretrained}_mrpc_new_epochs{number_epochs}",
        evaluation_strategy="epoch",
        learning_rate=2e-4,
        weight_decay=0.01,
        num_train_epochs=number_epochs,
        per_device_train_batch_size=32,
        per_device_eval_batch_size=32,
        logging_steps=1,
        # push_to_hub=True,
        # save_strategy="epoch",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=lm_dataset["train"],
        eval_dataset=lm_dataset["validation"],
        data_collator=data_collator,
    )

    if number_epochs > 0:
        trainer.train()

    eval_results = trainer.evaluate()
    print(f"Perplexity: {math.exp(eval_results['eval_loss']):.2f}")

    tokenizer.save_pretrained(training_args.output_dir, None, None, True)
    torch.save(model.state_dict(), 'mrpc_params.pth')


if __name__ == "__main__":
    set_seed(42)
    train()
