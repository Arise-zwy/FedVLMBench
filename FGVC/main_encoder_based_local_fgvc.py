import torch
import copy
import os
from tqdm import tqdm
import numpy as np
import logging
import datetime
import sys
import logging
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
sys.setrecursionlimit(2000)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args
from vlmmodel.vlm import init_vlm_model,make_supervised_data_module_local
import warnings
warnings.filterwarnings("ignore", category=UserWarning)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def compute_metrics(eval_pred):
   
    labels = eval_pred.label_ids
    preds = eval_pred.predictions[0]

    all_labels = []
    all_preds = []

    for case_labels, case_preds in zip(labels, preds):
        case_lab,case_pred=[],[]
        for label, pred in zip(case_labels, case_preds):
            if label != -100:
                case_lab.append(label)
                case_pred.append(pred)
        all_labels.append(tokenizer.decode(case_lab))
        all_preds.append(tokenizer.decode(case_pred))


    right=0
    all_num=0
    for label_sentence, pred_sentence in zip(all_labels, all_preds):
        label_is_pos=label_sentence.find("is")
        pred_is_pos=pred_sentence.find("is")
        if label_is_pos!=-1 and pred_is_pos!=-1:
            la=label_sentence[label_is_pos+3:-5]
            answer=pred_sentence[pred_is_pos+3:-5]
            # print("'"+la+"'","'"+answer+"'")
            if la==answer and la!=" ":
                right+=1
        all_num+=1
    print(right,all_num)
    accuracy=right/all_num

    return {
        "accuracy": accuracy,
    }

# ===== Define the arguments =====
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None)
args = parser.parse_args()
config_file=args.config
script_args, fed_args, peft_config, model_args, data_args = get_config(config_file)

training_args = get_training_args(script_args, script_args.learning_rate)
script_args = save_config(script_args, fed_args, model_args, data_args)


logdir = script_args.output_dir  
if not os.path.exists(logdir):
    os.makedirs(logdir)
logfile = os.path.join(logdir, 'training.log')
logging.basicConfig(filename=logfile, level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)


sys.stdout = StreamToLogger(logging.getLogger('STDOUT'), logging.INFO)
sys.stderr = StreamToLogger(logging.getLogger('STDERR'), logging.ERROR)
print(logdir)
with open("running.txt","a") as f:
    f.write(f"time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},current running file: {__file__}, logdir: {logdir}\n")



logging.info("cuda: %s",torch.cuda.is_available())
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Current memory allocated: {torch.cuda.max_memory_reserved(device) / (1024**3):.2f} GB")


try:
        
    ## ===== laod lamma =====
    model,tokenizer=init_vlm_model(script_args,model_args, data_args)
    # ===== Load the dataset =====
    local_datasets, global_test_dataset,data_collator = make_supervised_data_module_local(tokenizer=tokenizer, data_args=data_args,fed_args=fed_args) 


    # training_args.max_steps=len(local_datasets)//training_args.batch_size
    # script_args.max_steps=len(local_datasets)//training_args.batch_size
    # training_args.max_steps=100
    # script_args.max_steps=100
    logging.info("max_steps: %d",training_args.max_steps) 
    logging.info("batch_size: %d",training_args.batch_size) 

    # ===== Get model config =====
    device_map, quantization_config, torch_dtype = get_model_config(script_args)

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

    if training_args.gradient_checkpointing:
        model.enable_input_require_grads()
        
    resume_from_checkpoint=training_args.resume_from_checkpoint

    # ===== Define the tokenizer =====
    if tokenizer.pad_token is None:
        # tokenizer.pad_token = tokenizer.unk_token   # following vicuna
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # print(tokenizer.pad_token_id)
    # ===== Start federated training =====
    training_loss = []

    for round in tqdm(range(fed_args.num_rounds)):

        new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)      # manually schedule the learning rate
        training_args = get_training_args(script_args, new_lr)
        logging.info(f"round: {round}, new_lr: {new_lr}")

        # ===== Train local model on the client side =====
        trainer = get_local_vlm_sft_trainer(
            model=model,
            tokenizer=tokenizer,
            training_args=training_args,
            local_dataset=local_datasets,
            eval_dataset=None,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            fed_args=fed_args,
            script_args=script_args,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            resume_from_checkpoint=resume_from_checkpoint if round == 0 else None  # Only resume from checkpoint in the first round
        )
        logging.info(f"Current memory allocated: {torch.cuda.max_memory_reserved(device) / (1024**3):.2f} GB")

        results = trainer.train()
        logging.info(results)

        training_loss.append(results.training_loss)

    
        # ===== Save the model =====
        logging.info(">> ========================================eval ==============") 
        logging.info(">> ========================================eval ==============") 
        logging.info(">> ========================================eval ==============") 

    
        eval_results = trainer.evaluate(eval_dataset=global_test_dataset)
        logging.info(eval_results)
        with open(os.path.join(script_args.output_dir, "eval_results.txt"), "a") as f:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{current_time} - Round {round+1}: {eval_results}\n")
        

        if (round+1) % fed_args.save_model_freq == 0:
            import shutil
            logging.info(f"checkpoint-{round+1}")
            checkpoint_dir = os.path.join(script_args.output_dir, f"checkpoint-{round+1}")
            trainer.save_model(checkpoint_dir)

            max_checkpoints = 1
            checkpoints = [d for d in os.listdir(script_args.output_dir) if d.startswith("checkpoint-")]
            checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))

            if len(checkpoints) > max_checkpoints:
                for old_checkpoint in checkpoints[:-max_checkpoints]:
                    old_checkpoint_path = os.path.join(script_args.output_dir, old_checkpoint)
                    logging.info(f"Deleting old checkpoint: {old_checkpoint_path}")
                    if os.path.isdir(old_checkpoint_path):
                        shutil.rmtree(old_checkpoint_path)
        
        np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))
    with open("running.txt","a") as f:
        f.write(f"stop!!!! time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},current running file: {__file__}, logdir: {logdir}\n")
        f.write("======================================================\n")


except Exception as e:
    logging.error(e)
    logging.error("error")
    with open("running.txt","a") as f:
        f.write(f"stop!!!! time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},current running file: {__file__}, logdir: {logdir}\n")
        f.write("======================================================\n")

    raise e