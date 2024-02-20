# Merging stage of the model parameters

import torch

from transformers import (
    AutoConfig,
    AutoModel,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

def debug():
    # Load fine-tuned model state_dict
    finetuned_state_dict = torch.load("sst2_params.pth")

    # Load pre-trained model state_dict
    pretrained_state_dict = AutoModel.from_pretrained("distilgpt2").state_dict()

    # for key in pretrained_state_dict:
    #     if key not in finetuned_state_dict:
    #         print(key)
    
    for key in finetuned_state_dict:
        if key not in pretrained_state_dict:
            print(key)

def interpolate():
    # Load fine-tuned model state_dict
    finetuned_state_dict = torch.load("sst2_params.pth")

    # Load pre-trained model state_dict
    pretrained_state_dict = AutoModel.from_pretrained("distilgpt2").state_dict()

    # Linear Interpolation of weights 
    def linear_interpolation(model_1, model_2, alpha=0.5):
        merged_state_dict = {}
        for key in model_1:
            new_key = "transformer." + key
            if new_key in model_2:
                merged_state_dict[key] = alpha * model_1[key].to("cpu") + (1 - alpha) * model_2[new_key].to("cpu")
            else:
                merged_state_dict[key] = model_1[key]
        
        # for key in model_2:
        #     if key not in merged_state_dict:
        #         merged_state_dict[key] = model_2[key]
    
        return merged_state_dict

    config = AutoConfig.from_pretrained("distilgpt2")
    model = AutoModel.from_config(config)
    merged_state_dict = linear_interpolation(pretrained_state_dict, finetuned_state_dict)
    model.load_state_dict(merged_state_dict)
    save_location = "dumps/interpolated_pre_trained_fine_tuned_model"
    model.save_pretrained(save_location)

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.save_pretrained(save_location)

if __name__ == "__main__":
    interpolate()