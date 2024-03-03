import abc
import os

import torch
import torch.nn as nn
from functorch import jvp, make_functional_with_buffers
from transformers.modeling_outputs import (
    BaseModelOutputWithPastAndCrossAttentions,
    CausalLMOutputWithCrossAttentions,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutputWithPast,
    TokenClassifierOutput,
)
from transformers import AutoModelForCausalLM, AutoTokenizer

# from src.modeling import ImageEncoder
# from src.utils import DotDict


class LinearizedModel(nn.Module):
    """Creates a linearized version of a nn.Module.

    The linearized version of a model is a proper PyTorch model and can be
    trained as any other nn.Module.

    Args:
        model (nn.Module): The model to linearize. The trainable parameters of
            the linearized model will be initialized to the parameters of this
            model.
        init_model (nn.Module): A model of the same type as `model` containing
            the parameters around which the model is initialized. If not
            provided, `model` is used as the initialization model.
    """

    def __init__(self, model: nn.Module, init_model: nn.Module = None) -> None:
        """Initializes the linearized model."""
        super().__init__()
        if init_model is None:
            init_model = model

        func0, params0, self.buffers0 = make_functional_with_buffers(
            init_model.eval(), disable_autograd_tracking=True
        )
        self.func0 = lambda params, x: func0(params, self.buffers0, x)

        _, params, _ = make_functional_with_buffers(
            model, disable_autograd_tracking=True
        )

        self.params = nn.ParameterList(params)
        self.params0 = nn.ParameterList(params0)
        self._model_name = model.__class__.__name__

        # The intial parameters are not trainable.
        for p in self.params0:
            p.requires_grad = False

        # The params are.
        for p in self.params:
            p.requires_grad = True

    def __call__(self, x) -> torch.Tensor:
        """Computes the linearized model output using a first-order Taylor decomposition."""
        dparams = [p - p0 for p, p0 in zip(self.params, self.params0)]
        out, dp = jvp(
            lambda param: self.func0(param, x),
            (tuple(self.params0),),
            (tuple(dparams),),
        )
        return out + dp


# class LinearizedImageEncoder(abc.ABC, nn.Module):
#     """Creates a linearized version of an image encoder."""

#     def __init__(
#         self, args=None, keep_lang=False, image_encoder=None, init_encoder=None
#     ):
#         super().__init__()
#         if image_encoder is None:
#             image_encoder = ImageEncoder(args, keep_lang)
#         if init_encoder is None:
#             init_encoder = image_encoder

#         # Copy the attributes from the image encoder.
#         self.train_preprocess = image_encoder.train_preprocess
#         self.val_preprocess = image_encoder.val_preprocess
#         self.cache_dir = image_encoder.cache_dir

#         self._model_name = self._get_name(args.model)
#         self.model = LinearizedModel(init_model=init_encoder, model=image_encoder)

#     def _get_name(self, model_name):
#         if "__pretrained__" in model_name:
#             model_name, _ = model_name.split("__pretrained__", "")
#         return model_name

#     def forward(self, x):
#         # use the taylorized version of the model.
#         return self.model(x)

#     def __call__(self, x):
#         return self.forward(x)

#     def save(self, filename):
#         """Saves the linearized image encoder.

#         We save the model name in the state dict so that we can load the
#         correct model when loading the linearized image encoder. Directly using
#         torch.save would not work becuse func0 is not serializable.

#         Args:
#             filename (str): The path to save the taylorized image encoder.
#         """
#         if os.path.dirname(filename) != "":
#             os.makedirs(os.path.dirname(filename), exist_ok=True)

#         state_dict = self.state_dict()
#         state_dict["model_name"] = self._model_name

#         torch.save(state_dict, filename)

#     @classmethod
#     def load(cls, filename):
#         """Loads a linearized image encoder.

#         It first loads the state dict with the model name and then creates the
#         correct model and loads the state dict.

#         Args:
#             filename (str): The path to the taylorized image encoder.

#         Returns:
#             LinearizedImageEncoder: The loaded taylorized image encoder.
#         """
#         print(f"Loading image encoder from {filename}")
#         state_dict = torch.load(filename, map_location="cpu")

#         # ImageEncoder expects a DotDict
#         args = DotDict({"model": state_dict["model_name"]})
#         taylorized_encoder = cls(args)

#         # Remove the model name from the state dict so that we can load the
#         # model.
#         state_dict.pop("model_name")
#         taylorized_encoder.load_state_dict(state_dict)
#         return taylorized_encoder


class GPTWrapper(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def forward(self, x):
        input_ids: Optional[torch.LongTensor] = x.get("input_ids", None)
        past_key_values: Optional[Tuple[Tuple[torch.Tensor]]] = x.get(
            "past_key_values", None
        )
        attention_mask: Optional[torch.FloatTensor] = x.get("attention_mask", None)
        token_type_ids: Optional[torch.LongTensor] = x.get("token_type_ids", None)
        position_ids: Optional[torch.LongTensor] = x.get("position_ids", None)
        head_mask: Optional[torch.FloatTensor] = x.get("head_mask", None)
        inputs_embeds: Optional[torch.FloatTensor] = x.get("inputs_embeds", None)
        encoder_hidden_states: Optional[torch.Tensor] = x.get(
            "encoder_hidden_states", None
        )
        encoder_attention_mask: Optional[torch.FloatTensor] = x.get(
            "encoder_attention_mask", None
        )
        labels: Optional[torch.LongTensor] = x.get("labels", None)
        use_cache: Optional[bool] = x.get("use_cache", None)
        output_attentions: Optional[bool] = x.get("output_attentions", None)
        output_hidden_states: Optional[bool] = x.get("output_hidden_states", None)
        return_dict: Optional[bool] = x.get("return_dict", None)

        return self.model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        ).loss


class LinearizeWrapper(nn.Module):
    def __init__(self, model_name: str):
        super().__init__()
        model = GPTWrapper(model_name=model_name)
        self.model = LinearizedModel(model)

    def forward(self, **kwargs):
        dict_kwargs = {k: v for k, v in kwargs.items()}
        loss = self.model(dict_kwargs)
        return CausalLMOutputWithCrossAttentions(loss=loss)


if __name__ == "__main__":
    model_name = "distilgpt2"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    inputs["labels"] = inputs["input_ids"].clone()

    original_model = AutoModelForCausalLM.from_pretrained(model_name)
    outputs1 = original_model(**inputs)

    linearized_model = LinearizeWrapper(model_name)
    outputs2 = linearized_model(**inputs)

    print(
        f"outputs1.loss: {outputs1.loss}, outputs2.loss: {outputs2.loss}, they should be the same"
    )