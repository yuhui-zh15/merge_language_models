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
    sst_state_dict = torch.load("params/sst2_params.pth")

    # Load second fine-tuned model state_dict
    paws_state_dict = torch.load("params/paws_params.pth")

    # Load pre-trained model state_dict
    pretrained_state_dict = AutoModel.from_pretrained("distilgpt2").state_dict()

    # for key in pretrained_state_dict:
    #     new_key = "transformer." + key
    #     if new_key not in paws_state_dict:
    #         print(key)
    
    # for key in pretrained_state_dict:
    #     print(key)

    def linear_interpolation(model_1, model_2, alpha=0.8):
        merged_state_dict = {}
        for key in model_1:
            # if key == "lm_head.weight":
            #     merged_state_dict[key] = alpha * model_1[key].to("cpu") + (1 - alpha) * model_2[key].to("cpu")
            # else:
            if key != "lm_head.weight":
                new_key = key[12:]
                merged_state_dict[new_key] = alpha * model_1[key].to("cpu") + (1 - alpha) * model_2[key].to("cpu")
        
        return merged_state_dict
    
    merged_state_dict = linear_interpolation(sst_state_dict, paws_state_dict)
    for key in merged_state_dict:
        updated_key = "transformer." + key
        if not torch.equal(merged_state_dict[key].to("cpu"), paws_state_dict[updated_key].to("cpu")):
            print("differ")
        else:
            print("same")

def interpolate():
    # Load fine-tuned model state_dict
    sst_state_dict = torch.load("params/sst2_params.pth")

    # Load second fine-tuned model state_dict
    paws_state_dict = torch.load("params/paws_params.pth")

    # Linear Interpolation of weights 
    def linear_interpolation(model_1, model_2, alpha=0.8):
        merged_state_dict = {}
        for key in model_1:
            # if key == "lm_head.weight":
            #     merged_state_dict[key] = alpha * model_1[key].to("cpu") + (1 - alpha) * model_2[key].to("cpu")
            # else:
            if key != "lm_head.weight":
                new_key = key[12:]
                merged_state_dict[new_key] = alpha * model_1[key].to("cpu") + (1 - alpha) * model_2[key].to("cpu")
        
        # for key in model_2:
        #     if key not in merged_state_dict:
        #         merged_state_dict[key] = model_2[key]
    
        return merged_state_dict

    config = AutoConfig.from_pretrained("distilgpt2")
    model = AutoModel.from_config(config)
    merged_state_dict = linear_interpolation(sst_state_dict, paws_state_dict)
    model.load_state_dict(merged_state_dict)
    save_location = "dumps/interpolated_fine_tuned_model_0.8_new"
    model.save_pretrained(save_location)

    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    tokenizer.save_pretrained(save_location)

if __name__ == "__main__":
    interpolate()