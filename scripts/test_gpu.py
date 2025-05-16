
# #模型下载
# # from modelscope import snapshot_download
# # model_dir = snapshot_download('AI-ModelScope/clip-vit-large-patch14-336')
# import torch
# from pynvml import *
# import time
# import sys

# import smtplib
# from email.mime.text import MIMEText
# from email.mime.multipart import MIMEMultipart

# # 发件人和收件人信息
# sender_email = "你的邮箱@gmail.com"
# receiver_email = "收件人邮箱@example.com"
# password = "你的邮箱密码"

# def send_email(content):
#     # 创建邮件
#     message = MIMEMultipart()
#     message["From"] = sender_email
#     message["To"] = receiver_email
#     message["Subject"] = "自动发送邮件示例"

#     # 添加邮件正文
#     message.attach(MIMEText(content, "plain"))

#     # 连接到SMTP服务器
#     with smtplib.SMTP("smtp.gmail.com", 587) as server:
#         server.starttls()
#         server.login(sender_email, password)
#         server.sendmail(sender_email, receiver_email, message.as_string())

#     print("邮件已发送成功！")


# # count传GPU个数，threshold是阈值，低于此阈值说明GPU是空闲的，second是每几秒进行继续轮训
# def select_gpu(count=torch.cuda.device_count(), threshold=1024, second=5):
#     nvmlInit()
#     if count == 0:
#         return 'cpu'
#     # 需要在多个GPU轮训找出空闲的GPU
#     current = 0
#     while True:
#         # 检查当前GPU是否可用
#         handle = nvmlDeviceGetHandleByIndex(current)
#         info = nvmlDeviceGetMemoryInfo(handle)
#         used_memory = info.used // (1024 * 1024)
#         if used_memory < threshold:  # 如果刺入小于阈值的内存，那么说明此GPU并没有被占用，可抢占
#             sys.stderr.write(
#                 f'此时GPU{current}使用内存为[{used_memory}MB]，低于阈值[{threshold}]才可抢占----GPU{current}可抢占，将抢占GPU：{current}号GPU----{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n')
#             send_email((f'此时GPU{current}使用内存为[{used_memory}MB]，低于阈值[{threshold}]才可抢占----GPU{current}不可抢占，继续轮训----{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n'))
#             break
#             nvmlShutdown()
#             return current
#         else:
#             print("here")
#             send_email((f'此时GPU{current}使用内存为[{used_memory}MB]，低于阈值[{threshold}]才可抢占----GPU{current}不可抢占，继续轮训----{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n'))
#             sys.stderr.write(
#                 f'此时GPU{current}使用内存为[{used_memory}MB]，低于阈值[{threshold}]才可抢占----GPU{current}不可抢占，继续轮训----{time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())}\n')
#         time.sleep(second)
#         current = (current + 1) % count
#         break
# select_gpu()


import torch
from transformers import BertModel, BertTokenizer
import copy
import os
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DataCollatorForCompletionOnlyLM
from peft import get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict, prepare_model_for_kbit_training

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args
from vlmmodel.vlm import init_vlm_model,make_supervised_data_module,make_supervised_data_module_clients
import vlmmodel.vlm
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

##
# init_vlm_model() sets up the types of the vision tower, language model and tokenizer
# make_supervised_data_module() sets up the dataset and dataloader

# ===== Define the arguments =====
script_args, fed_args, peft_config, model_args, data_args = get_config()
training_args = get_training_args(script_args, script_args.learning_rate)
save_config(script_args, fed_args, model_args, data_args)
# print(script_args, fed_args)

## laod lamma 这里需要加入visual encoder
model,tokenizer=vlmmodel.vlm.init_vlm_model_test(script_args,model_args, data_args)


# 将模型移动到GPU（如果可用的话）
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# 显示当前模型占用的显存量
print(f"Current memory allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB")

# 显示模型在整个过程中最大占用的显存量
print(f"Peak memory allocated: {torch.cuda.max_memory_allocated(device) / (1024**3):.2f} GB")

# 打印显存使用情况摘要
print(torch.cuda.memory_summary())
import torchinfo
torchinfo.summary(model)


