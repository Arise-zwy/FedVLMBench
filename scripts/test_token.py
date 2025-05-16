import os
import copy
import json
import logging
import pathlib
import transformers
from PIL import Image
from dataclasses import dataclass, field
from vlmmodel.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN


 
tokenizer = transformers.AutoTokenizer.from_pretrained(
            "/disk1/Data/Medical-Ours/Brain_Tumor_data/ckpt/llama-3-8B/Meta-Llama-3___1-8B",
            model_max_length=1024,
            padding_side="right",
            use_fast=False,
        )
print(tokenizer.encode("is A"))
print(tokenizer.encode("is B"),tokenizer.decode(tokenizer.encode("is B")))
print(tokenizer.encode("is C"),tokenizer.decode(tokenizer.encode("is C")))
print(tokenizer.encode("is D"),tokenizer.decode(tokenizer.encode("is D")))

