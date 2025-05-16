import os
import copy
import json
import logging
import pathlib
import transformers
from PIL import Image
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List
import torch
from torch.utils.data import Dataset
from vlmmodel.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
# from vlmmodel.train.trainer import VLMTrainer
from vlmmodel import conversation as conversation_lib
from vlmmodel.model.mobilellama import MobileLlamaForCausalLM
from vlmmodel.utils import tokenizer_image_token
# from datasets import Dataset
from config import DataArguments
from transformers.trainer import (ALL_LAYERNORM_LAYERS,
                                  get_parameter_names)
from peft import get_peft_model_state_dict, set_peft_model_state_dict
import pandas as pd
import glob

from datasets import load_dataset
from vlmmodel.model.fullmodel import PureLlamaForCausalLM
local_rank = None 

def rank0_print(*args):
    if local_rank == 0:
        print(*args)



def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler']

    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        # weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.state_dict().items(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    # print("has_image",has_image)
    # print("conversation",conversation)
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}  # {'human': 'USER', 'gpt': 'ASSISTANT'}

    # Apply prompt templates
    #0211 把选项换成别的

    option_dicts = {}
    option_dicts[0]={"(A)":"(1)","(B)":"(2)","(C)":"(3)","(D)":"(4)"}
    option_dicts[1]={"(A)":"(a)","(B)":"(b)","(C)":"(c)","(D)":"(d)"}
    option_dicts[2]={"(A)":"(e)","(B)":"(f)","(C)":"(g)","(D)":"(h)"}
    option_dicts[3]={"(A)":"(E)","(B)":"(F)","(C)":"(G)","(D)":"(H)"}

    answer_dicts = {}
    answer_dicts[0]={"A":"1","B":"2","C":"3","D":"4"}
    answer_dicts[1]={"A":"a","B":"b","C":"c","D":"d"}
    answer_dicts[2]={"A":"e","B":"f","C":"g","D":"h"}
    answer_dicts[3]={"A":"E","B":"F","C":"G","D":"H"}

    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        # random select a label_dict
        label_dict_index = torch.randint(0,4,(1,)).item()
        label_dict = option_dicts[label_dict_index]
        answer_dict = answer_dicts[label_dict_index]
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            # if role == "USER":
            #     for key in label_dict.keys():
            #         sentence["value"] = sentence["value"].replace(key,label_dict[key])
            # else:
            #     for key in answer_dict.keys():
            #         sentence["value"] = sentence["value"].replace(key,answer_dict[key])
            conv.append_message(role, sentence["value"])
          
        conversations.append(conv.get_prompt())
        # print(conversations)

    # Tokenize conversations
    # print("conversation",conversations)
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "  # ' ASSISTANT: '
    for conversation, target in zip(conversations, targets):
        # print(target,len(target))
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
      
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)

            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))-1 # rou前面的128000去掉
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
       
        # print("====================================")
        cur_len+=3 # 《/S》
        target[cur_len:] = IGNORE_INDEX
        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                    f"======================="
                )
    # print(targets)
    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep
            round_len = len(tokenizer_image_token(rou, tokenizer)) + len(tokenizer_image_token(conv.sep, tokenizer))
            instruction_len = len(tokenizer_image_token(parts[0], tokenizer))
            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep  # \n
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))  # 2
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets)



def preprocess_vqa(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}  # {'human': 'USER', 'gpt': 'ASSISTANT'}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            # Skip the first one if it is not from human
            source = source[1:]
        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "  # ' ASSISTANT: '
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 2
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
           
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    # print(targets)
    return dict(
        input_ids=input_ids,
        labels=targets,
    )




def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        # print("preprocess_plain")
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        # print("preprocess_llama_2")
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        # print("preprocess_v1")
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        # print("preprocess_mpt")
        return preprocess_mpt(sources, tokenizer)
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)

# change the class type to datasets
# class LazySupervisedDataset(Dataset):
class LazySupervisedDataset(Dataset):

    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 subarea="",
                 labels=["label"], 
                 data_len=0,
                 ):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))
        self.image_folder = data_args.image_folder
      

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        if "SciQA_caption" in data_args.dataset_name:
            self.conver_name="conversations_caption"
        elif "SciQA_vqa" in data_args.dataset_name:
            self.conver_name="conversations_vqa"
        else:
            self.conver_name="conversations"

        # self.dataset_name=self.image_folder.split("/")[-4]
        self.dataset_name = data_args.dataset_name
        
        if subarea!="":
            new_list_data_dict = []
            for data in self.list_data_dict:
                exp_label= data[subarea]
                if exp_label in labels:
                    new_list_data_dict.append(data)
            self.list_data_dict = new_list_data_dict
        if data_len!=0:
            self.list_data_dict = list_data_dict[:data_len]
        
        self.data_args = data_args
       

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample[self.conver_name]) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample[self.conver_name])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
       
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.image_folder
            processor = self.data_args.image_processor
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e[self.conver_name] for e in sources]), self.data_args)
        else:
            sources = copy.deepcopy([e[self.conver_name] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict

    def select(self, indices):
        # 选择特定的样本索引
        self.data = [self.list_data_dict[i] for i in indices]





@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)

def make_supervised_data_module_clients(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                fed_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
   
    # client_training_datasets=[]
    # for i in range(fed_args.num_clients):
    #     ds = load_dataset('json', data_files=data_args.data_path+"train/client_"+str(i)+'.json')
    #     client_training_datasets.append(ds)
    
    # global_test_dataset = load_dataset('json', data_files=data_args.data_path+"test/test.json")

    # return dict(client_training_datasets=client_training_datasets,
    #             global_test_dataset=global_test_dataset)

    client_training_datasets=[]
    client_files = glob.glob(data_args.data_path + "train/client_*.json")
    print(f"Number of client files: {len(client_files)}")
    print(fed_args.num_clients)
    if len(client_files) > fed_args.num_clients:
        raise ValueError("Number of client files is more than the number of clients.")

    for i in range(fed_args.num_clients):
       
        print(data_args.data_path+"train/client_"+str(i)+'.json')
        ds = LazySupervisedDataset(data_path=data_args.data_path+"train/client_"+str(i)+'.json',tokenizer=tokenizer,
                                data_args=data_args)
        client_training_datasets.append(ds)

    
    
    global_test_dataset = LazySupervisedDataset(data_path=data_args.data_path+"test/test.json",tokenizer=tokenizer,
                                data_args=data_args)
    
    subarea_test_datasets = []
    # if fed_args.test_local:
    #     subarea="topic"
    #     if subarea=="topic":
    #         subarea_labels = ["topic1","topic2","topic3","topic4","topic5","topic6","topic7","topic8","topic9","topic10"]
    #     elif subarea=="sentiment":
    #         subarea_labels = ["positive","negative","neutral"]
    #     elif subarea=="emotion":
    #         subarea_labels = ["joy","sadness","fear","anger","disgust","surprise"]
    #     else:
    #         subarea_labels = ["label"]
    #     for label in subarea_labels:
    #         ds = LazySupervisedDataset(data_path=data_args.data_path+"test/test.json",tokenizer=tokenizer,
    #                                 data_args=data_args,subarea=subarea,labels=label)
    #         subarea_test_datasets.append(ds)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return client_training_datasets,global_test_dataset,subarea_test_datasets, data_collator


def make_supervised_data_module_clients_non_iid(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                fed_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
 

    client_training_datasets=[]
    client_files = glob.glob(data_args.data_path + "train/client_*.json")
    print(f"Number of client files: {len(client_files)}")
    if len(client_files) < fed_args.num_clients:
        raise ValueError("Number of client files is less than the number of clients.")
    
    for i in range(fed_args.num_clients):
        print(data_args.data_path+"train/client_"+str(i)+'.json')
        ds = LazySupervisedDataset(data_path=data_args.data_path+"train/client_"+str(i)+'.json',tokenizer=tokenizer,
                                data_args=data_args)
        client_training_datasets.append(ds)

    
    
    global_test_dataset = LazySupervisedDataset(data_path=data_args.data_path+"test/test.json",tokenizer=tokenizer,
                                data_args=data_args)
    

    subarea_test_datasets = []
  

    if data_args.dataset_name == "Scivqa" or data_args.dataset_name == "sci_caption":
        if "subject" in data_args.data_path:
            subarea="subject"
            subarea_labels = [['social science'], ['natural science'], ['language science']]
        elif "sub_topic" in data_args.data_path:
            # ['science-and-engineering-practices', 'chemistry', 'us-history','economics', 'writing-strategies']
            subarea="topic"
            subarea_labels =[["science-and-engineering-practices"], ["chemistry"], ["us-history"], ["economics"], ["writing-strategies"]]
        elif "topic" in data_args.data_path:
            subarea="topic"
            subarea_labels = [['chemistry', 'science-and-engineering-practices', 'earth-science'],['biology'],['physics'],['geography'],['writing-strategies', 'economics', 'us-history']]
        else:
            subarea=""
            subarea_labels = [""]
    elif data_args.dataset_name == "SLAKE":
        if "subject" in data_args.data_path:
            subarea="subject"
            subarea_labels = [['social science'], ['natural science'], ['language science']]
        elif "sub_topic" in data_args.data_path:
            # ['science-and-engineering-practices', 'chemistry', 'us-history','economics', 'writing-strategies']
            subarea="topic"
            subarea_labels =[["science-and-engineering-practices"], ["chemistry"], ["us-history"], ["economics"], ["writing-strategies"]]
        elif "topic" in data_args.data_path:
            subarea="topic"
            subarea_labels = [['chemistry', 'science-and-engineering-practices', 'earth-science'],['biology'],['physics'],['geography'],['writing-strategies', 'economics', 'us-history']]
        else:
            subarea=""
            subarea_labels = [""]
    else:
        KeyError("subarea not found")
    
    # if subarea!="":
    #     for label in subarea_labels:
    #         print("get subarea",subarea, label)
    #         ds = LazySupervisedDataset(data_path=data_args.data_path+"test/test.json",tokenizer=tokenizer,
    #                                 data_args=data_args,subarea=subarea,labels=label)
    #         subarea_test_datasets.append(ds)

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return client_training_datasets,global_test_dataset,subarea_test_datasets, data_collator


def make_supervised_data_module_mix_data_clients_iid(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                fed_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
 
    client_training_datasets=[]
    client_files = glob.glob(data_args.data_path + "train/client_*.json")
    print(f"Number of client files: {len(client_files)}")
    if len(client_files) < fed_args.num_clients:
        raise ValueError("Number of client files is less than the number of clients.")
    for i in range(fed_args.num_clients):
        print(data_args.data_path+"train/client_"+str(i)+'.json')
        ds = LazySupervisedDataset(data_path=data_args.data_path+"train/client_"+str(i)+'.json',tokenizer=tokenizer,
                                data_args=data_args)
        client_training_datasets.append(ds)

    
    global_test_datasets=[]

    
    test_files = glob.glob(data_args.data_path + "test/*.json")
    if len(test_files) == 1:
        global_test_datasets.append(LazySupervisedDataset(data_args.data_path+"test/test.json",tokenizer=tokenizer,
                                data_args=data_args))
    else:
        for test_file in test_files:
            print("test_file",test_file)
            if "/test.json" not in test_file:

                global_test_datasets.append(LazySupervisedDataset(test_file,tokenizer=tokenizer,
                                data_args=data_args))
                
    # global_test_datasets.append(LazySupervisedDataset(data_args.data_path+"test/caption_test.json",tokenizer=tokenizer,
    #                             data_args=data_args))
    # global_test_datasets.append(LazySupervisedDataset(data_args.data_path+"test/vqa_test.json",tokenizer=tokenizer,
    #                             data_args=data_args))    
    subarea_test_datasets = []
    print("len global_test_datasets",len(global_test_datasets))

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return client_training_datasets,global_test_datasets,subarea_test_datasets, data_collator



def make_supervised_data_module_mix_local(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                fed_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
   
    # print("train file", data_args.data_path + "train/train.json")
    # training_datasets = LazySupervisedDataset(data_path=data_args.data_path+"train/train.json",tokenizer=tokenizer,
    #                             data_args=data_args)
    
    if data_args.data_path.endswith(".json"):
        training_datasets =  LazySupervisedDataset(data_path=data_args.data_path,tokenizer=tokenizer,
                                data_args=data_args)
        data_args.data_path = data_args.data_path.split("train")[0]
        print("rewrite data_dir",data_args.data_path)
        
    else:
        training_datasets =  LazySupervisedDataset(data_path=data_args.data_path+"train/train.json",tokenizer=tokenizer,
                                data_args=data_args)
    
    global_test_datasets=[]

    
    test_files = glob.glob(data_args.data_path + "test/*.json")
    if len(test_files) == 1:
        global_test_datasets.append(LazySupervisedDataset(data_args.data_path+"test/test.json",tokenizer=tokenizer,
                                data_args=data_args))
    else:
        for test_file in test_files:
            print("test_file",test_file)
            if "/test.json" not in test_file:

                global_test_datasets.append(LazySupervisedDataset(test_file,tokenizer=tokenizer,
                                data_args=data_args))
                  
    subarea_test_datasets = []
    print("len global_test_datasets",len(global_test_datasets))

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return training_datasets,global_test_datasets, data_collator


def make_supervised_data_module_local(tokenizer: transformers.PreTrainedTokenizer,
                                config) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
   
    data_dir=config.dataset.data_path
    
    if data_dir.endswith(".json"):
        training_datasets =  LazySupervisedDataset(data_path=data_dir,tokenizer=tokenizer,
                                data_args=config.dataset)
        data_dir = data_dir.split("train")[0]
        print("rewrite data_dir",data_dir)
        
    else:
        training_datasets =  LazySupervisedDataset(data_path=data_dir+"train/train.json",tokenizer=tokenizer,
                                data_args=config.dataset)
    
    global_test_datasets=[]
    test_files = glob.glob(data_dir + "test/*.json")
    if len(test_files) == 1:
        global_test_datasets = LazySupervisedDataset(data_dir+"test/test.json",tokenizer=tokenizer,
                                data_args=config.dataset)
    else:
        for test_file in test_files:
            print("test_file",test_file)
            if "/test.json" not in test_file:

                global_test_datasets.append(LazySupervisedDataset(test_file,tokenizer=tokenizer,
                                data_args=config.dataset))
                
    
        print("len global_test_datasets",len(global_test_datasets))

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return training_datasets,global_test_datasets,data_collator

def make_supervised_data_module_local(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                fed_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
   
   
    training_datasets = LazySupervisedDataset(data_path=data_args.data_path+"train/train.json",tokenizer=tokenizer,
                                data_args=data_args)
    
    global_test_dataset = LazySupervisedDataset(data_path=data_args.data_path+"test/test.json",tokenizer=tokenizer,
                                data_args=data_args)
    
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return training_datasets,global_test_dataset,data_collator



def make_supervised_data_module_test(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                ) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
  
    global_test_dataset = LazySupervisedDataset(data_path=data_args.data_path+"test/test.json",tokenizer=tokenizer,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return global_test_dataset,data_collator






def init_vlm_model(training_args,model_args,data_args):
    def save_trainable_parameters(model, file_path=training_args.output_dir+"/trainable_parameters.csv"):
        params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params.append({"Parameter": name, "Shape": param.shape, "Requires Grad": param.requires_grad})
        
        df = pd.DataFrame(params)
        df.to_csv(file_path, index=False)
        print(f"Trainable parameters saved to {file_path}")

   
       
    global local_rank

    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            quantization_config=quantization_config
        
        ))



    print("load model")
    if model_args.vision_tower is not None:
        model = PureLlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            **bnb_model_from_pretrained_args,
        )
    
    model.config.use_cache = False

    print("training_args.bit ",training_args.bits)
    if training_args.bits in [4, 8]:
        print("training_args.bits in [4, 8]")
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    print(find_all_linear_names(model))
    from peft import LoraConfig, get_peft_model
    lora_config = LoraConfig(
        r=training_args.peft_lora_r,
        lora_alpha=training_args.peft_lora_alpha,
        target_modules=find_all_linear_names(model),
        lora_dropout=training_args.lora_dropout,
        bias=training_args.lora_bias,
        task_type="CAUSAL_LM",
    )
    if training_args.bits == 16:
        if training_args.bf16:
            model.to(torch.bfloat16)
        if training_args.fp16:
            model.to(torch.float16)
    rank0_print("Adding LoRA adapters...")
    print("Adding LoRA adapters...")
    model = get_peft_model(model, lora_config)


    ## judge whether using fast tokenizer
    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )
    print("Loading tokenizer from", model_args.model_name_or_path)

    ## judge the type of the pad token
    if model_args.pad_token_version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.pad_token_version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.pad_token_version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.pad_token_version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    # add special tokens to the tokenizer

    
    special_token_list = [
    conversation_lib.default_conversation.roles[0] + ": ",
    conversation_lib.default_conversation.roles[1] + ": ",
    ".",
    ]

    tokenizer.add_special_tokens({'additional_special_tokens': special_token_list})
    # If there is a mismatch between tokenizervocab size and embedding matrix,self.llama model.base model.model.mode1ding matrix

    if model.config.vocab_size != tokenizer.vocab_size:
        model.resize_token_embeddings(len(tokenizer))

    # print the special tokens of the tokenizer
    print("Special tokens:", tokenizer.special_tokens_map)


    if model_args.vision_tower is not None:
        model.initialize_vision_modules(model_args=model_args, fsdp=training_args.fsdp)  
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        model.config.vision_tower_type = training_args.vision_tower_type = model_args.vision_tower_type


        if training_args.bits in [4, 8]:
            model.mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    
    print("print_trainable_parameters")
    model.model.requires_grad_(False)
    if training_args.lora_enable:
        for name, param in model.named_parameters():
            if 'lora' in name and "mm" not in name:
                param.requires_grad = True
        model.print_trainable_parameters()

    if model_args.tune_mm_mlp_adapter:
        # model.requires_grad_(False)
        for p in model.mm_projector.parameters():
            p.requires_grad = True

    save_trainable_parameters(model)



    return model, tokenizer

    # data_module = make_supervised_data_module(tokenizer=tokenizer, data_args=data_args)
    # trainer = VLMTrainer(model=model, tokenizer=tokenizer, args=training_args, **data_module)

    # if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    #     trainer.train(resume_from_checkpoint=True)
    # else:
    #     trainer.train()
    # trainer.save_state()

    # model.config.use_cache = True

    # if training_args.lora_enable:
    #     state_dict = get_peft_state_maybe_zero_3(
    #         model.named_parameters(), training_args.lora_bias
    #     )
    #     non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
    #         model.named_parameters()
    #     )
    #     if training_args.local_rank == 0 or training_args.local_rank == -1:
    #         model.config.save_pretrained(training_args.output_dir)
    #         model.save_pretrained(training_args.output_dir, state_dict=state_dict)
    #         print('non_lora_trainable...', non_lora_state_dict.keys())
    #         torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    # else:
    #     safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)


def init_vlm_model_test(testing_args,model_args,data_args):
    tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            model_max_length=testing_args.model_max_length,
            padding_side="right",
            use_fast=True,
        )
    print("Loading tokenizer from", model_args.model_name_or_path)
    ## judge the type of the pad token
    if model_args.pad_token_version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.pad_token_version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.pad_token_version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.pad_token_version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]
    special_token_list = [
    conversation_lib.default_conversation.roles[0] + ": ",
    conversation_lib.default_conversation.roles[1] + ": ",
    ".",
    ]

    tokenizer.add_special_tokens({'additional_special_tokens': special_token_list})
    print("Special tokens:", tokenizer.special_tokens_map)


    model = PureLlamaForCausalLM.load_merge_model(
        model_args.model_name_or_path,
        model_args.lora_path,
        len(tokenizer)
    )
    print("Loading model from", model_args.model_name_or_path)
    print("Loading lora from", model_args.lora_path)
    # model = PureLlamaForCausalLM.from_pretrained(
    #     model_args.model_name_or_path,
    # )

    model.config.use_cache = True


    if model.config.vocab_size != tokenizer.vocab_size:
        model.resize_token_embeddings(len(tokenizer))


    if model_args.vision_tower is not None:
        model.initialize_vision_modules(model_args=model_args, fsdp=testing_args.fsdp)  
        
        vision_tower = model.get_vision_tower()

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.image_grid_pinpoints = data_args.image_grid_pinpoints

        model.config.vision_tower_type = testing_args.vision_tower_type = model_args.vision_tower_type

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        testing_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

    return model, tokenizer





def set_train_pro_mode(model, training_args,model_args,script_args):

    model.model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

   
    if model_args.vision_tower is not None:
        
        tune_mm_mlp_adapter=True
        if tune_mm_mlp_adapter:
            for p in model.mm_projector.parameters():
                p.requires_grad = True

      

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    
    print("print_trainable_parameters")
    model.print_trainable_parameters()
    def save_trainable_parameters(model, file_path=script_args.output_dir+"/1_stage_trainable_parameters.csv"):
        params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params.append({"Parameter": name, "Shape": param.shape, "Requires Grad": param.requires_grad})
        
        df = pd.DataFrame(params)
        df.to_csv(file_path, index=False)
        print(f"Trainable parameters saved to {file_path}")

    save_trainable_parameters(model)
    
    return model


def set_train_lora_mode(model, training_args,model_args,script_args):
    model.requires_grad_(False)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    ## lora enable
    lora_enable=True
    if lora_enable:
        for name, param in model.named_parameters():
            if 'lora' in name:
                param.requires_grad = True



    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    
    print("print_trainable_parameters")
    model.print_trainable_parameters()

    def save_trainable_parameters(model, file_path=script_args.output_dir+"/2_stage_trainable_parameters.csv"):
        params = []
        for name, param in model.named_parameters():
            if param.requires_grad:
                params.append({"Parameter": name, "Shape": param.shape, "Requires Grad": param.requires_grad})
        
        df = pd.DataFrame(params)
        df.to_csv(file_path, index=False)
        print(f"Trainable parameters saved to {file_path}")

    save_trainable_parameters(model)
    
    return model


def get_selected_state_dict(model, avg_lora=False, avg_pro=False):
    state_dict = {}
    if avg_pro:
        projercor_params = get_projecter_state_dict(model)
        # for name, param in projercor_params.items():
        #     projercor_params[name] = param.cpu()
        state_dict.update(projercor_params)
        
    if avg_lora:
        fed_params = get_peft_model_state_dict(model)
        # for name, param in fed_params.items():
        #     fed_params[name] = param.cpu()

        state_dict.update(fed_params)
            
        
    return state_dict


def set_selected_state_dict(model,global_dict, lora_enable=False, projercor_enable=False):
    if projercor_enable:
        set_projecter_state_dict(model, global_dict)
    if lora_enable:
        set_peft_model_state_dict(model, global_dict)
        

def get_projecter_state_dict(model):
    state_dict = {}
    decay_parameters = get_parameter_names(
                model, ALL_LAYERNORM_LAYERS)
    # decay_parameters = [
    #             name for name in decay_parameters if "bias" not in name]
    for name, param in model.named_parameters():
        if 'mm_projector' in name and name in decay_parameters and param.requires_grad:
            state_dict[name] = param
        
    return state_dict


def set_projecter_state_dict(model, global_dict):
  
    projecter_state_dict = {k: v.to('cuda') for k, v in global_dict.items() if "mm_projector" in k}
    # print(projecter_state_dict.keys())

    model_state_dict = model.state_dict()

    model_state_dict.update(projecter_state_dict)

    model.load_state_dict(model_state_dict)

def print_trainable_parameters(model):
    try:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Parameter: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
        
        print("Exiting print_trainable_parameters")
    except Exception as e:
        print(f"An error occurred in print_trainable_parameters: {e}")
