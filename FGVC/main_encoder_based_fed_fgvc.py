import torch
import copy
import os
from tqdm import tqdm
import numpy as np
import logging
import warnings
import datetime
import sys
import time
import argparse
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore", category=UserWarning)
sys.setrecursionlimit(3000)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils import *
from federated_learning import *
from config import get_config, save_config, get_model_config, get_training_args
from vlmmodel.vlm import init_vlm_model,make_supervised_data_module_clients,set_selected_state_dict,get_selected_state_dict,make_supervised_data_module_clients_non_iid


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
        import random
        if random.random()<0.05:
            print("label:"+label_sentence+"==> pred:"+pred_sentence)
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
import shutil
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None)
args = parser.parse_args()
config_file=args.config
script_args, fed_args, peft_config, model_args, data_args = get_config(config_file)
training_args = get_training_args(script_args, script_args.learning_rate)
script_args = save_config(script_args, fed_args, model_args, data_args)
set_seed(script_args.seed)

# logging.info(f"config_file: {config_file}")
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
writer = SummaryWriter(log_dir=logdir+'/vis')

with open("running_med.txt","a") as f:
    f.write(f"time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},current running file: {__file__}, config: {config_file} ,logdir: {logdir}\n")

logging.info("cuda: %s",torch.cuda.is_available())
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Current memory allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB")
logging.info("===============================================================")
logging.info("===============================================================")
logging.info("===============================================================")

# logging.info("avg_lora: %s",fed_args.avg_lora)
# logging.info("avg_projector: %s",fed_args.avg_projector)

if fed_args.avg_lora and fed_args.avg_projector and script_args.lora_enable and model_args.tune_mm_mlp_adapter:
    logging.info("all_avg !!!!!!!!!!!!!!!!")
elif  fed_args.avg_lora and not fed_args.avg_projector and  script_args.lora_enable and not model_args.tune_mm_mlp_adapter:
    logging.info("only_lora !!!!!!!!!!!!!!!!")
elif  fed_args.avg_lora and not fed_args.avg_projector and  script_args.lora_enable and  model_args.tune_mm_mlp_adapter:
    logging.info("share_lora local pro !!!!!!!!!!!!!!!!")
elif  not fed_args.avg_lora and  fed_args.avg_projector and  script_args.lora_enable and  model_args.tune_mm_mlp_adapter:
    logging.info("share_pro local lora  !!!!!!!!!!!!!!!!")
elif fed_args.avg_projector and model_args.tune_mm_mlp_adapter and not fed_args.avg_lora and not script_args.lora_enable:
    logging.info("only_pro !!!!!!!!!!!!!!!!")
else:
    logging.info("no_avg !!!!!!!!!!!!!!!!")
logging.info("test_local: %s",fed_args.test_local)
logging.info("max_steps: %d",training_args.max_steps) 
logging.info("batch_size: %d",training_args.batch_size) 


try:
    ## ===== laod lamma =====
    model,tokenizer=init_vlm_model(script_args,model_args, data_args)

    if data_args.dataset_name=="FGVC":
        if "variant" in data_args.data_path:
            fed_args.num_clients=10
            fed_args.sample_clients=10
            print("variant")
        elif "family" in data_args.data_path:
            fed_args.num_clients=10
            fed_args.sample_clients=10
            print("family")
        elif "manufacturer" in data_args.data_path:
            fed_args.num_clients=10
            fed_args.sample_clients=10
            print("manufacturer")
        else:
            ValueError("dataset error")
    if data_args.dataset_name=="FGVC":
        if "5clients" in data_args.data_path:
            fed_args.num_clients=5
            fed_args.sample_clients=5
            print("variant")
        else:
            ValueError("dataset error")

    # ===== Load the dataset =====
    if "non" in fed_args.split_strategy:
        local_datasets, global_test_dataset,sub_test_datasets,data_collator = make_supervised_data_module_clients_non_iid(tokenizer=tokenizer, data_args=data_args,fed_args=fed_args) 

    else:
        local_datasets, global_test_dataset,sub_test_datasets,data_collator = make_supervised_data_module_clients(tokenizer=tokenizer, data_args=data_args,fed_args=fed_args) 
    # local_datasets, global_test_dataset,data_collator = make_supervised_data_module_clients(tokenizer=tokenizer, data_args=data_args,fed_args=fed_args) 
    sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]
    logging.info("sample_num_list: %s",sample_num_list) 


    # ===== Get model config =====

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    resume_from_checkpoint=training_args.resume_from_checkpoint
    select_keys=["eval_accuracy","eval_loss"]

    # ===== Define the global and local models =====
    share_params= get_selected_state_dict(model,avg_lora=fed_args.avg_lora,avg_pro=fed_args.avg_projector)
    global_dict = copy.deepcopy(share_params)
    upload_local_dict_list = [copy.deepcopy(global_dict) for _ in range(fed_args.num_clients)]
    
    if fed_args.fed_alg in ['fedprox','scaffold','fedadagrad', 'fedyogi', 'fedadam',"fedavgm"]:
        for key in global_dict.keys():
            global_dict[key] = global_dict[key].to(device)


    local_params= get_selected_state_dict(model,avg_lora=not fed_args.avg_lora,avg_pro=not fed_args.avg_projector)
    save_local_list = [copy.deepcopy(local_params) for _ in range(fed_args.num_clients)]
    logging.info("lora_enable: %s,projercor_enable %s",fed_args.avg_lora,fed_args.avg_projector)
    
    logging.info("global_dict.keys(): %s", list(global_dict.keys())[0:10]+list(global_dict.keys())[-10:])
    logging.info("local_params.keys(): %s", local_params.keys())

    proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
    global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)

    # ===== Define the tokenizer =====
    if tokenizer.pad_token is None:
        # tokenizer.pad_token = tokenizer.unk_token   # following vicuna
        tokenizer.pad_token = tokenizer.pad_token or tokenizer.eos_token

    # ===== Start federated training =====
    training_loss = [[] for i in range(fed_args.num_clients)]
    test_metrics = [{} for i in range(fed_args.num_clients)]

    for round in tqdm(range(fed_args.num_rounds)):
        t_start=time.time()
        clients_this_round = get_clients_this_round(fed_args, round)

        logging.info(f">> ==================== Round {round+1} : {clients_this_round} ====================")
        logging.info(f">> ==================== Round {round+1} : {clients_this_round} ====================")
        logging.info(f">> ==================== Round {round+1} : {clients_this_round} ====================")
    
        for client in range(fed_args.num_clients):
            if client not in clients_this_round:
                training_loss[client].append(-1)            # -1 is an indicator of not training
                continue
            logging.info(f">> ============: {client} ==============")
            set_selected_state_dict(model, global_dict, lora_enable=fed_args.avg_lora,projercor_enable=fed_args.avg_projector)  
            set_selected_state_dict(model, save_local_list[client], lora_enable=not fed_args.avg_lora,projercor_enable=not fed_args.avg_projector)

            new_lr = cosine_learning_rate(round, fed_args.num_rounds, script_args.learning_rate, 1e-6)      # manually schedule the learning rate
            training_args = get_training_args(script_args, new_lr)
            logging.info(f"round: {round}, new_lr: {new_lr}")

            # ===== Train local model on the client side =====
            trainer = get_fed_local_vlm_sft_trainer(
                model=model,
                tokenizer=tokenizer,
                training_args=training_args,
                local_dataset=local_datasets[client],
                compute_metrics=compute_metrics,
                data_collator=data_collator,
                global_dict=global_dict,
                fed_args=fed_args,
                script_args=script_args,
                local_auxiliary=auxiliary_model_list[client],
                global_auxiliary=global_auxiliary,
                preprocess_logits_for_metrics=preprocess_logits_for_metrics,
                eval_dataset=None,
                resume_from_checkpoint=resume_from_checkpoint if round == 0 else None  # Only resume from checkpoint in the first round
            )
        
            results = trainer.train()
            logging.info(f"Current memory allocated: {torch.cuda.max_memory_reserved(device) / (1024**3):.2f} GB")

            training_loss[client].append(results.training_loss)
            logging.info(results)

            # ===== Client transmits local information to server =====
            if fed_args.fed_alg == 'scaffold':
                auxiliary_model_list[client], auxiliary_delta_dict[client] = trainer.get_auxiliary_param()

            # upload_local_dict_list[client] = copy.deepcopy(get_peft_model_state_dict(model))   # copy is needed!
            upload_params = get_selected_state_dict(model,avg_lora=fed_args.avg_lora,avg_pro=fed_args.avg_projector)
            upload_local_dict_list[client] = copy.deepcopy(upload_params)
            
            save_local_params=get_selected_state_dict(model,avg_lora=not fed_args.avg_lora,avg_pro=not fed_args.avg_projector)
            save_local_list[client] = copy.deepcopy(save_local_params)  



            # if fed_args.test_local and round % 3 == 0 and round >=0 :
            #     print("local eval")
            #     for idx,subdata in enumerate(sub_test_datasets):
            #         eval_results = trainer.evaluate(eval_dataset=subdata)
            #         with open(os.path.join(script_args.output_dir, "eval_results.txt"), "a") as f:
            #             f.write(f"    ---------------- Clinet{client} : subdataset{idx}, {eval_results}\n")
            #     eval_results = trainer.evaluate(eval_dataset=global_test_dataset)
            #     with open(os.path.join(script_args.output_dir, "eval_results.txt"), "a") as f:
            #         f.write(f"     - Round {round+1} Clinet{client} : {eval_results}\n")
            #         f.write("================================\n")
                                
            #     for key,value in eval_results.items():
            #         if key in select_keys:
            #             test_metrics[client][key] = eval_results[key]

            torch.cuda.empty_cache()  
    
        # ===== summary loss =====
        avg_loss = np.mean([training_loss[i][-1] for i in range(fed_args.num_clients) if i in clients_this_round])
        writer.add_scalar('Train/Average Training Loss', avg_loss, round + 1)
        if fed_args.test_local and round % 3 == 0 and round > 5:
            for key in select_keys:
                key_values = [test_metrics[client][key] for client in range(fed_args.num_clients) if key in test_metrics[client]]
                if key_values:
                    avg_key, std_key = compute_avg_and_std(key_values)
                    logging.info(f"Round {round+1}, Key: {key}, Avg: {avg_key}, Std: {std_key}")
                    writer.add_scalar(f"local_test_{key}/Avg", avg_key, round + 1)
                    writer.add_scalar(f"local_test_{key}/Std", std_key, round + 1)
                    with open(os.path.join(script_args.output_dir, "eval_results.txt"), "a") as f:
                        f.write(f"    summary - Round {round+1} : {key}, Avg: {avg_key}, Std: {std_key}\n")
  
        # ===== Server aggregates the local models =====
        use_cpu=False
        if fed_args.num_clients>5:
            use_cpu=True
        global_dict, global_auxiliary = global_aggregate(
            fed_args, global_dict, upload_local_dict_list, sample_num_list, \
            clients_this_round, round, proxy_dict=proxy_dict, \
            opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict),use_cpu=use_cpu
        )
        set_selected_state_dict(model, global_dict, lora_enable=fed_args.avg_lora,projercor_enable=fed_args.avg_projector)
        torch.cuda.empty_cache()  

        # ===== Test the model =====
        logging.info(">> ========================================eval ==============") 
        logging.info(">> ========================================eval ==============") 
        logging.info(">> ========================================eval ==============") 

        eval_results = trainer.evaluate(eval_dataset=global_test_dataset)
        logging.info(eval_results)
        for key in select_keys:
            writer.add_scalar(f"Global_test/Average {key}", eval_results[key], round + 1)
        with open(os.path.join(script_args.output_dir, "eval_results.txt"), "a") as f:
            current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            f.write(f"{current_time} - Round {round+1}: {eval_results}\n")
        if "non" in fed_args.split_strategy:
            for idx,subdata in enumerate(sub_test_datasets):
                eval_results = trainer.evaluate(eval_dataset=subdata)
                with open(os.path.join(script_args.output_dir, "eval_results.txt"), "a") as f:
                    f.write(f"    ---------------- subdataset{idx}, {eval_results}\n")
            with open(os.path.join(script_args.output_dir, "eval_results.txt"), "a") as f:
                f.write("================================\n")
                f.write("================================\n")

        # if (round+1) % fed_args.save_model_freq == 0:
        #     logging.info(f"checkpoint-{round+1}")
        #     checkpoint_dir = os.path.join(script_args.output_dir, f"checkpoint-{round+1}")
        #     trainer.save_model(checkpoint_dir)

        #     max_checkpoints = 1
        #     checkpoints = [d for d in os.listdir(script_args.output_dir) if d.startswith("checkpoint-")]
        #     checkpoints = sorted(checkpoints, key=lambda x: int(x.split('-')[1]))

        #     if len(checkpoints) > max_checkpoints:
        #         for old_checkpoint in checkpoints[:-max_checkpoints]:
        #             old_checkpoint_path = os.path.join(script_args.output_dir, old_checkpoint)
        #             logging.info(f"Deleting old checkpoint: {old_checkpoint_path}")
        #             if os.path.isdir(old_checkpoint_path):
        #                 shutil.rmtree(old_checkpoint_path)
        
        np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))
        t_end=time.time()
        logging.info(f"round time: {(t_end-t_start)/3600:.2f} hours")

    with open("running_med.txt","a") as f:
        f.write(f"stop!!!! time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},current running file: {__file__}, logdir: {logdir}\n")
        f.write("======================================================\n")


except Exception as e:
    logging.error(e)
    logging.error("error")
    with open("running_med.txt","a") as f:
        f.write(f"stop!!!! time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},current running file: {__file__}, logdir: {logdir}\n")
        f.write("======================================================\n")
    raise e