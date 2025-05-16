from typing import List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers import PreTrainedModel, PretrainedConfig


import torch

from transformers.utils import ModelOutput
from dataclasses import dataclass
from showomodel.prompting_utils import create_attention_mask_for_mmu

@dataclass
class CausalLMOutputWithPast(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    # past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    # hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    # attentions: Optional[Tuple[torch.FloatTensor, ...]] = None
    labels: Optional[Tuple[torch.FloatTensor, ...]] = None


class VLMConfig(PretrainedConfig):
    model_type = "mobilevlm"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
     


from showomodel.modeling_showo import Showo
from showomodel.modeling_magvitv2 import MAGVITv2


def get_vq_model_class(model_type):
    if model_type == "magvitv2":
        return MAGVITv2
    # elif model_type == "vq16":
    #     return VQ_16
    else:
        raise ValueError(f"model_type {model_type} not supported.")


class Full_showo(PreTrainedModel):
    config_class = VLMConfig

    def __init__(self, config):
        super().__init__(config)
         # VQ model for processing image into discrete tokens
        # self.vq_model = get_vq_model_class(config.model.vq_model.type)
        self.vq_model = None
        # self.config = config
        self.model = None
        self.uni_prompting=None
        self.max_seq_length = 1024
        # self.gradient_checkpointing = False
        # self.config = config

    def get_model(self):
        return self.model
    
    def get_vq_model(self):
        return self.vq_model
    
    def set_prompting(self, prompting):
        self.uni_prompting = prompting

    def gradient_checkpointing_enable(self, **kwargs):
        self.gradient_checkpointing = True
        if hasattr(self.model, "gradient_checkpointing_enable"):
            self.model.gradient_checkpointing_enable(**kwargs)
        print(f"{self.__class__.__name__}: Gradient checkpointing enabled.")

    def gradient_checkpointing_disable(self):
        self.gradient_checkpointing = False
        if hasattr(self.model, "gradient_checkpointing_disable"):
            self.model.gradient_checkpointing_disable()
        print(f"{self.__class__.__name__}: Gradient checkpointing disabled.")


    @classmethod
    def from_pretrained(cls, config ,**kwargs):
        model_config = VLMConfig()
        model = cls(model_config)

        model.vq_model = get_vq_model_class("magvitv2")
        print("loading pretrained model from huggingface: pretrain_model/magvitv2",)
        print(config.model.vq_model.vq_model_name)
        model.vq_model = model.vq_model.from_pretrained(config.model.vq_model.vq_model_name)
        
        
        vocab_size = config.model.showo.vocab_size
        model.max_seq_length = config.dataset.preprocessing.max_seq_length
        print(config.model.showo.pretrained_model_path)
        print("Loading model from", config.model.showo.pretrained_model_path)
        model.model = Showo.from_pretrained(config.model.showo.pretrained_model_path)
        if vocab_size != model.model.vocab_size:
            model.model.showo.resize_token_embeddings(vocab_size) 
            model.model.config.codebook_size = 8192
            model.model.config.vocab_size = vocab_size
            model.model.vocab_size = vocab_size
            model.model.output_size = vocab_size
            model.model.config.mask_token_id = vocab_size - 1
            model.model.mask_token_id = vocab_size - 1
        return model




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
        
        # output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        # output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        # return_dict = return_dict if return_dict is not None else self.config.use_return_dict 
        return_dict = return_dict 
      

        attention_mask_mmu, past_key_values, input_ids_mmu, labels_mmu= self.prepare_inputs_labels_for_multimodal(input_ids,labels,images, past_key_values)
        
        # import pdb; pdb.set_trace()
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
     
        # print(f"input_ids_mmu shape: {input_ids_mmu.shape}")
        # print(f"attention_mask_mmu shape: {attention_mask_mmu.shape}")
        # print(f"labels_mmu shape: {labels_mmu.shape}")
        

        logits,_,_, loss = self.model(
                    input_ids=input_ids_mmu,
                    input_embeddings=None,
                    attention_mask=attention_mask_mmu,
                    labels=labels_mmu,
                    max_seq_length=self.max_seq_length,
                )

        # if not return_dict:
        #     output = (logits,)
        #     return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            # past_key_values=outputs.past_key_values,
            # hidden_states=outputs.hidden_states,
            # attentions=outputs.attentions,
            labels=labels_mmu[..., 1:]
        )

    def prepare_inputs_labels_for_multimodal(
        self, input_ids, labels, images, past_key_values):
    
        pixel_values_mmu, input_ids_mmu, labels_mmu = (images,input_ids,labels)

        image_tokens_mmu = self.vq_model.get_code(pixel_values_mmu)
        image_tokens_mmu = image_tokens_mmu + len(self.uni_prompting.text_tokenizer)
      
        input_ids_mmu = torch.cat([
            (torch.ones(input_ids_mmu.shape[0], 1).cuda() * self.uni_prompting.sptids_dict['<|mmu|>'].cuda()),
            (torch.ones(input_ids_mmu.shape[0], 1).cuda() * self.uni_prompting.sptids_dict['<|soi|>'].cuda()),
            image_tokens_mmu,
            (torch.ones(input_ids_mmu.shape[0], 1).cuda() * self.uni_prompting.sptids_dict['<|eoi|>'].cuda()),
            input_ids_mmu,
        ], dim=1).long()

        labels_mmu = torch.cat([
            (torch.ones(input_ids_mmu.shape[0], 1).cuda() * self.uni_prompting.ignore_id),
            (torch.ones(input_ids_mmu.shape[0], 1).cuda() * self.uni_prompting.ignore_id),
            torch.ones_like(image_tokens_mmu) * self.uni_prompting.ignore_id,
            (torch.ones(input_ids_mmu.shape[0], 1).cuda() * self.uni_prompting.ignore_id),
            labels_mmu,
        ], dim=1).long()


        # print(input_ids_mmu[0])
        # # print(labels_mmu[0])
        # decode_labels=[]
        # for label in (input_ids_mmu[0]):
        #     if label != -100:
        #         decode_labels.append(label)
        # print(decode_labels)
        # print(self.uni_prompting.text_tokenizer.decode(decode_labels))

        # print("=====================================")



        attention_mask_mmu =create_attention_mask_for_mmu(sequence=input_ids_mmu,
                                                        eoi_id=int(self.uni_prompting.sptids_dict['<|eoi|>']))
        
        mask_dtype = self.model.showo.model.embed_tokens.weight.dtype
        attention_mask_mmu = attention_mask_mmu.to(mask_dtype)
  
        return attention_mask_mmu, past_key_values, input_ids_mmu, labels_mmu
    


# AutoConfig.register("mobilevlm", VLMConfig)
AutoModelForCausalLM.register(VLMConfig, Full_showo)
