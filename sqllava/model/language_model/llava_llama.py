


from typing import List, Optional, Tuple, Union

import numpy as np
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from llava.model.WD_loss import WassersteinDisloss
from torch.nn.functional import normalize

class LlavaConfig(LlamaConfig):
    model_type = "llava"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


def plot_dist(input, target, loss):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from scipy.interpolate import make_interp_spline
    N,D = input.shape
    input_ = input.detach().cpu().float().numpy()
    target_ = target.detach().cpu().float().numpy()
    x=np.linspace(0,D,D)

    for i in range(len(input_[:2])):
        plt.figure(figsize=(8,6))
        X_ = np.linspace(x.min(), x.max(), 500)
        X_Y_Spline1 = make_interp_spline(x, input_[i])
        X_Y_Spline2 = make_interp_spline(x, target_[i])
        Y1 = X_Y_Spline1(X_)
        Y2 = X_Y_Spline2(X_)
        plt.plot(Y1, label="input distribution")
        plt.plot(Y2, label="target distribution")
        plt.legend()
        plt.grid(True)
        plt.savefig(f'./figures/{loss}_{i}_dis.pdf', format='pdf', bbox_inches='tight')
        plt.close()

class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig
    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()
        try:
            self.qavloss = config.qav_loss
            self.loss_alpha = config.loss_alpha
        except:
            self.qavloss = False
            self.loss_alpha=0.
    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        try:
            input_ids, attention_mask, past_key_values, inputs_embeds, labels, image_features = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
        except:
            input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
                input_ids, attention_mask, past_key_values, labels, images)

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )
        hidden_states = outputs[0]

        logits = self.lm_head(hidden_states)
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
            #print("va loss: ",loss)

        if self.qavloss:
            n, l, d = image_features.shape
            #loss_fct = nn.KLDivLoss(reduction="batchmean", log_target=True)
            loss_fct = nn.MSELoss()
            #image_features=torch.mean(image_features,dim=1,keepdim=True)
            #v = image_features.contiguous().view(-1, d)
            v_llm = hidden_states[:, -l - 2:-2, :]
            image_features = torch.mean(image_features,dim=1)
            v_llm = torch.mean(v_llm,dim=1)
            #inp = F.log_softmax(image_features,dim=-1)
            #target = F.log_softmax(v_llm,dim=-1)

            qavloss = loss_fct(image_features, v_llm)
            plot_dist(image_features, v_llm, qavloss)
            loss = loss + self.loss_alpha*qavloss

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
                "images": kwargs.get("images", None),
            }
        )
        return model_inputs

AutoConfig.register("llava", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM)
