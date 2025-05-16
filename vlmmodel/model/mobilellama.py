from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from vlmmodel.model.mobilevlm import MobileVLMMetaModel, MobileVLMMetaForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig, AutoModel


class MobileVLMConfig(LlamaConfig):
     model_type = "mobilevlm"
    #  def __init__(self,  **kwargs):
    #     super().__init__(**kwargs)

    #     self.vision_tower_type = "clip"
    #     self.vision_tower = "/home/weiying/Documents/models--openai--clip-vit-base-patch32/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
    #     self.mm_projector_type="linear"
    #     self.mm_vision_select_layer=-2
    #     self.mm_use_im_start_end=False
    #     self.mm_use_im_patch_token=False


# MobileVLMMetaModel build the vision tower, llamamodel build the language model
class MobileLlamaModel(MobileVLMMetaModel, LlamaModel):
    config_class = MobileVLMConfig

    def __init__(self, config: MobileVLMConfig):

        super(MobileLlamaModel, self).__init__(config)

# MobileVLMMetaForCausalLM encode the image and text for the language model
class MobileLlamaForCausalLM(LlamaForCausalLM, MobileVLMMetaForCausalLM):
    config_class = MobileVLMConfig 

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        # 必要的话就更换为llama
        self.model = MobileLlamaModel(config)
        # self.model = LlamaForCausalLM(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()  # Initialize weights and apply final processing

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
        output_hidden_states = (output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states)
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict 

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = \
            self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)

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

from vlmmodel.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, \
    DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

def load_pretrained_model(model_path, model_base, script_args, model_args, data_args, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):


    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.float16

    tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
    # ===== Define the tokenizer =====
    if tokenizer.pad_token is None:
    # tokenizer.pad_token = tokenizer.unk_token   # following vicuna
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token
    base_model = init_vlm_model(script_args,model_args,data_args,model_base, **kwargs)

    # base_model = MobileLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)
    vision_encoder = AutoModel.from_pretrained("/home/weiying/Documents/models--openai--clip-vit-base-patch32/3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268")
    base_model.load_state_dict(vision_encoder.state_dict(), strict=False) 

    from peft import PeftModel

    lora_model = PeftModel.from_pretrained(
        base_model,
        model_path,
        torch_dtype=torch.float16,
    )
 
    # print("Applying the LoRA")
    model = lora_model.merge_and_unload()

    # for name,parm in vision_encoder.named_parameters():
    #     print(name)
    # for name,parm in model.named_parameters():
    #     print(name)

    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    vision_tower = model.get_vision_tower()
    if 'v2' in getattr(model.config, "mm_projector_type", "ldpnet"):
        vision_tower.load_image_processor()
    elif not vision_tower.is_loaded:
        vision_tower.load_model()
    vision_tower.to(device=device, dtype=torch.float16)
    image_processor = vision_tower.image_processor
    model.to(device=device, dtype=torch.float16)

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048
    
    return tokenizer, model, image_processor, context_len



def init_vlm_model(training_args,model_args,data_args,model_base, **kwargs):


    # from vlmmodel.model.fullmodel import PureLlamaForCausalLM
    # model = PureLlamaForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    #     cache_dir=training_args.cache_dir,
    # )
   
    model = MobileLlamaForCausalLM.from_pretrained(model_base, low_cpu_mem_usage=True, **kwargs)

    model.config.use_cache = False

    model.get_model().initialize_vision_modules(model_args=model_args)  
    
    # vision_tower = model.get_vision_tower()
    # vision_tower.to(device=training_args.device)

    # data_args.image_processor = vision_tower.image_processor
    # data_args.is_multimodal = True

    model.config.image_aspect_ratio = data_args.image_aspect_ratio
    model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

    model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
    model.config.vision_tower_type = training_args.vision_tower_type = model_args.vision_tower_type


    model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter


    model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
    model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    model.config.mm_projector_lr = training_args.mm_projector_lr

    
    return model

# AutoConfig.register("mobilevlm", MobileVLMConfig)
AutoModelForCausalLM.register(MobileVLMConfig, MobileLlamaForCausalLM)
