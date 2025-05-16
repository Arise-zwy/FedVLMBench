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
from showomodel import conversation as conversation_lib
from showomodel.prompting_utils import UniversalPrompting

# from datasets import Dataset
from config import DataArguments

import pandas as pd
import glob

from datasets import load_dataset
from showomodel.model.fullmodel import Full_showo
local_rank = None 

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler',"vq_model"]
    # print("find_all_linear_names")
    for name, module in model.named_modules():
        # print(name,"vq_model" in name)
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])


    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

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


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments) -> Dict:
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()

                # Customized operation, get rid of <image> special token. Edited by Zechen
                sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "")
                sentence['value'] = sentence['value'].strip()

    return sources


def preprocess_v0(
        sources,
        tokenizer,
        has_image: bool = False
):
    # Let's assume has_image is false, since we will process the image token separately
    has_image = False

    # Adapted from llava-phi/mipha/train/train.py
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
            assert role == conv.roles[j % 2]
            conv.append_message(role, sentence["value"])
        conversation_str = str(conv.get_prompt()).strip()
        conversations.append(conversation_str)

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
    sep = conv.sep + conv.roles[1] + ": "                   # ' ASSISTANT: '
    for conversation, target in zip(conversations, targets):        # loop for instances in a batch
        # total_len = int(target.ne(tokenizer.pad_token_id).sum()) + conversation.count(conv.sep2)  # in phi-2, pad_token_id == eos_token_id
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)              # handle multi-round conversation regarding one image
        cur_len = 0                                         # no bos token in phi, so set the initial len to 0
        if cur_len > 0:
            target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            round_len = len(tokenizer(rou).input_ids) + 1  # +1 for <|endoftext|>
            instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            target[cur_len: cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len


        target[cur_len:] = IGNORE_INDEX
      
     

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(conversation)
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )
    SYSTEM_PROMPT = "A chat between a curious user and an artificial intelligence assistant. " \
                "The assistant gives helpful, detailed, and polite answers to the user's questions."

    input_ids_system = tokenizer(
        [SYSTEM_PROMPT for _ in range(len(conversations))],
        return_tensors="pt",
        padding="longest",
        max_length=tokenizer.model_max_length,
        truncation=True,
    ).input_ids
  


    return dict(
        input_ids=input_ids,
        labels=targets,
        input_ids_system=input_ids_system
    )


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
        print("data_args.dataset_name",data_args.dataset_name)
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

        assert 'image' in sources[0]
        image_file = self.list_data_dict[i]['image']
        image_folder = self.image_folder
        try:
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            def image_transform(image, resolution=256, normalize=True):
                from torchvision import transforms
                image = transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC)(image)
                image = transforms.CenterCrop((resolution, resolution))(image)
                image = transforms.ToTensor()(image)
                if normalize:
                    image = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)(image)
                return image
            image = image_transform(image)
        except:
            print("Read image error. Use dummy data.")
            crop_size = 256
            image = torch.zeros(3, crop_size, crop_size)

        sources = preprocess_multimodal(copy.deepcopy([e[self.conver_name] for e in sources]), self.data_args)

        # data_dict = preprocess_v0(sources, self.tokenizer)

        data_dict = preprocess_v0(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))

        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0],
                             input_ids_system=data_dict["input_ids_system"][0]
                             )

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        else:
            # image does not exist in the data, but the model is multimodal
            crop_size = 256
            data_dict['image'] = torch.zeros(3, crop_size, crop_size)

        return data_dict

    def select(self, indices):
        # 选择特定的样本索引
        self.data = [self.list_data_dict[i] for i in indices]





@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels,input_ids_system = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels", "input_ids_system"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        # max_length = 512+28
            
        # input_ids_system = torch.stack(input_ids_system, dim=0)
        # offset = max_length - input_ids.shape[-1] - input_ids_system.shape[-1]

        # if input_ids.shape[-1] < max_length - input_ids_system.shape[-1]:
        #     pad_tube = torch.ones(size=(input_ids.shape[0], offset), dtype=input_ids.dtype) * self.tokenizer.pad_token_id
        #     input_ids = torch.cat([input_ids, pad_tube], dim=1)

        #     pad_tube = torch.ones(size=(labels.shape[0], offset), dtype=labels.dtype) * IGNORE_INDEX
        #     labels = torch.cat([labels, pad_tube], dim=1)

        # min_max_len = min(
        #     max_length - input_ids_system.shape[-1],
        #     self.tokenizer.model_max_length - input_ids_system.shape[-1],
        # )

        # input_ids = input_ids[:, :min_max_len]
        # labels = labels[:, :min_max_len]


        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        # print(input_ids[0])
        # print(labels[0])
        # decode_labels=[]
        # for label in (input_ids[0]):
        #     if label != -100:
        #         decode_labels.append(label)
        # print(self.tokenizer.decode(decode_labels))

        # print("=====================================")
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            # attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
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
                                config) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
   
 
    num_clients = config.fed.num_clients
    data_dir=config.dataset.data_path
    client_training_datasets=[]
    client_files = glob.glob(data_dir+ "train/client_*.json")
    print(f"Number of client files: {len(client_files)}")
    print(num_clients)
    if len(client_files) > num_clients:
        raise ValueError("Number of client files is more than the number of clients.")

    for i in range(num_clients):
       
        print(data_dir+"train/client_"+str(i)+'.json')
        ds = LazySupervisedDataset(data_path=data_dir+"train/client_"+str(i)+'.json',tokenizer=tokenizer,
                                data_args=config.dataset)
        client_training_datasets.append(ds)

        global_test_datasets=[]

    
    test_files = glob.glob(data_dir + "test/*.json")
    if len(test_files) == 1:
        global_test_datasets=LazySupervisedDataset(data_dir+"test/test.json",tokenizer=tokenizer,
                                data_args=config.dataset)
    else:
        for test_file in test_files:
            print("test_file",test_file)
            if "/test.json" not in test_file:

                global_test_datasets.append(LazySupervisedDataset(test_file,tokenizer=tokenizer,
                                data_args=config.dataset))
                
    
    subarea_test_datasets = []
    print("len global_test_datasets",len(global_test_datasets))

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return client_training_datasets,global_test_datasets,subarea_test_datasets, data_collator



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



def make_supervised_data_module_test(tokenizer: transformers.PreTrainedTokenizer,
                                data_args,
                                ) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
  
    global_test_dataset = LazySupervisedDataset(data_path=data_args.data_path+"test/test.json",tokenizer=tokenizer,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    return global_test_dataset,data_collator






def init_vlm_model(config):
    def save_trainable_parameters(model, file_path=config.experiment.output_dir+"/trainable_parameters.csv"):
        params = []
        for name, param in model.model.named_parameters():
            if param.requires_grad:
                params.append({"Parameter": name, "Shape": param.shape, "Requires Grad": param.requires_grad})
        
        df = pd.DataFrame(params)
        df.to_csv(file_path, index=False)
        print(f"Trainable parameters saved to {file_path}")

    
    global local_rank
  
  
   
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model.showo.llm_model_path,
        padding_side="left",
    )
    print("Loading tokenizer from", config.model.showo.llm_model_path)
    uni_prompting = UniversalPrompting(tokenizer,max_text_len=1024,
                                    #    max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=(
                                           "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                                           "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
                                       ),
                                       ignore_id=-100, cond_dropout_prob=0.1)

    print('special tokens : \n', uni_prompting.sptids_dict)
    
    print("training_args.bit ",config.fed.bits)

    
    print("load model")
    
   
    model = Full_showo.from_pretrained(config)
    model.set_prompting(uni_prompting)


    # print("print_trainable_parameters")

    model.vq_model.requires_grad_(False)
    model.model.requires_grad_(True)

    save_trainable_parameters(model)



    return model, tokenizer



def init_vlm_model_lora(config):
    def save_trainable_parameters(model, file_path=config.experiment.output_dir+"/trainable_parameters.csv"):
        params = []
        for name, param in model.model.named_parameters():
            if param.requires_grad:
                params.append({"Parameter": name, "Shape": param.shape, "Requires Grad": param.requires_grad})
        
        df = pd.DataFrame(params)
        df.to_csv(file_path, index=False)
        print(f"Trainable parameters saved to {file_path}")

    
    global local_rank
  
  
   
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        config.model.showo.llm_model_path,
        padding_side="left",
    )
    print("Loading tokenizer from", config.model.showo.llm_model_path)
    uni_prompting = UniversalPrompting(tokenizer,max_text_len=1024,
                                    #    max_text_len=config.dataset.preprocessing.max_seq_length,
                                       special_tokens=(
                                           "<|soi|>", "<|eoi|>", "<|sov|>", "<|eov|>", "<|t2i|>",
                                           "<|mmu|>", "<|t2v|>", "<|v2v|>", "<|lvg|>"
                                       ),
                                       ignore_id=-100, cond_dropout_prob=0.1)

    print('special tokens : \n', uni_prompting.sptids_dict)
    
    print("training_args.bit ",config.fed.bits)

    
    print("load model")
    
   
    model = Full_showo.from_pretrained(config)
    model.set_prompting(uni_prompting)

    from peft import LoraConfig, get_peft_model
    # lora_config = LoraConfig(
    #     r=training_args.peft_lora_r,
    #     lora_alpha=training_args.peft_lora_alpha,
    #     target_modules=find_all_linear_names(model),
    #     lora_dropout=training_args.lora_dropout,
    #     bias=training_args.lora_bias,
    #     task_type="CAUSAL_LM",
    # )
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=find_all_linear_names(model),
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    rank0_print("Adding LoRA adapters...")
    print("Adding LoRA adapters...")
    model = get_peft_model(model, lora_config)


    # print("print_trainable_parameters")
    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True

 

    save_trainable_parameters(model)



    return model, tokenizer



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


    model = Full_showo.load_merge_model(
        model_args.model_name_or_path,
        model_args.lora_path,
        len(tokenizer)
    )
    print("Loading model from", model_args.model_name_or_path)
    print("Loading lora from", model_args.lora_path)
    # model = Full_showo.from_pretrained(
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



def set_selected_state_dict(model,global_dict):
    model.model.load_state_dict(global_dict, strict=False)


from peft import get_peft_model_state_dict, set_peft_model_state_dict

def get_selected_state_dict_lora(model):
    state_dict = {}
    fed_params = get_peft_model_state_dict(model)
    state_dict.update(fed_params)
            
        
    return state_dict


def set_selected_state_dict_lora(model,global_dict):
    set_peft_model_state_dict(model, global_dict)
        
  

    
def print_trainable_parameters(model):
    try:
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(f"Parameter: {name}, Shape: {param.shape}, Requires Grad: {param.requires_grad}")
        
        print("Exiting print_trainable_parameters")
    except Exception as e:
        print(f"An error occurred in print_trainable_parameters: {e}")
