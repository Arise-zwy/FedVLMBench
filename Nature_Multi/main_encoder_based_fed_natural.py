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
import glob
from torch.utils.tensorboard import SummaryWriter
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

sys.setrecursionlimit(3000)
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from utils import *
from federated_learning import *
from config import get_config, save_config, get_training_args
from vlmmodel.vlm import init_vlm_model,make_supervised_data_module_clients,make_supervised_data_module_clients_non_iid,set_selected_state_dict,get_selected_state_dict,make_supervised_data_module_mix_data_clients_iid
from pycocoevalcap import compute_scores
import random

def compute_acc_metrics(eval_pred):
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

    if all_labels[0].find("caption")!=-1:
        gts={}
        res={}
        iid=0
    
        for label_sentence, pred_sentence in zip(all_labels, all_preds):          
            label_is_pos = label_sentence.find("is")
            pred_is_pos = pred_sentence.find("is")
            if label_is_pos != -1 and pred_is_pos != -1:
                gts[iid]=[label_sentence[label_is_pos + 4:-5]]
                res[iid]=[pred_sentence[pred_is_pos + 4:-5]]
                iid+=1
                if iid==10:
                    print(gts,res)
        result=compute_scores(gts, res, method="English_word", tokenizer=None, temp=True)
        print("len:",len(gts))
        return {
            "CIDEr": result["CIDEr"],
            "ROUGE_L": result["ROUGE_L"],
        }    


    elif all_labels[0].find("classification")!=-1:
        right=0
        all_num=0
        for label_sentence, pred_sentence in zip(all_labels, all_preds):
            if random.random()<0.005:
                print("label:",label_sentence)
                print("pred:",pred_sentence)
            label_is_pos=label_sentence.find("is")
            pred_is_pos=pred_sentence.find("is")
            if label_is_pos!=-1 and pred_is_pos!=-1:
                la=label_sentence[label_is_pos+3:label_is_pos+4]
                answer=pred_sentence[pred_is_pos+3:pred_is_pos+4]
                if la==answer and la!=" ":
                    right+=1
            all_num+=1

        accuracy=right/all_num
        print(right,all_num,accuracy)

        return {
            "class_accuracy": accuracy,
        }
    elif all_labels[0].find("question")!=-1:
        right=0
        all_num=0
        for label_sentence, pred_sentence in zip(all_labels, all_preds):
            if random.random()<0.005:
                print("label:",label_sentence)
                print("pred:",pred_sentence)
            label_is_pos=label_sentence.find("is")
            pred_is_pos=pred_sentence.find("is")
            if label_is_pos!=-1 and pred_is_pos!=-1:
                la=label_sentence[label_is_pos+3:label_is_pos+4]
                answer=pred_sentence[pred_is_pos+3:pred_is_pos+4]
                if la==answer and la!=" ":
                    right+=1
            all_num+=1

        accuracy=right/all_num
        print(right,all_num,accuracy)

        return {
            "vqa_accuracy": accuracy,
        }
    elif all_labels[0].find("coordinates")!=-1:
        print(all_preds[0:5])
        iou_scores = []
        len_ok=0
        for label_sentence, pred_sentence in zip(all_labels, all_preds):
            label_is_pos = label_sentence.find("are")
            pred_is_pos = pred_sentence.find("are")
            if label_is_pos != -1 and pred_is_pos != -1:
                len_ok+=1
                label_pos = label_sentence[label_is_pos + 3:-5]
                answer_pos = pred_sentence[pred_is_pos + 3:-5]
              
                if label_pos and answer_pos and is_valid_box_format(label_pos) and is_valid_box_format(answer_pos):
                    import ast
                    label_boxes = ast.literal_eval(label_pos)
                    try:
                        pred_boxes = ast.literal_eval(answer_pos)
                    except ValueError as e:
                        print(f"Error parsing answer_pos: {answer_pos}")
                        pred_boxes = None  

                    if pred_boxes != None and len(label_boxes) == 4 and len(pred_boxes) == 4:
                        label_boxes = [label_boxes]
                        pred_boxes = [pred_boxes]
                        for label_box, pred_box in zip(label_boxes, pred_boxes):
                            # print("label_boxes:",label_box,type(label_box))
                            # print("pred_boxes:",pred_box,type(pred_box))

                            if isinstance(label_box, dict) and isinstance(pred_box, dict):
                                for key in label_box:
                                    if key in pred_box:
                                        iou = compute_iou(label_box[key], pred_box[key])
                                        iou_scores.append(iou)
                                        if random.random() < 0.05:
                                            print("label_box:",label_box)
                                            print("pred_box:",pred_box)
                                            print("iou:",iou)
                                            print("=====================================")
                            elif isinstance(label_box, list) and isinstance(pred_box, list):
                                iou = compute_iou(label_box, pred_box)
                                iou_scores.append(iou)
                                if random.random() < 0.05:
                                    print("label_box:",label_box)
                                    print("pred_box:",pred_box)
                                    print("iou:",iou)
                                    print("=====================================")

        mean_iou = sum(iou_scores) / len(iou_scores) if iou_scores else 0.0
        print("len:",len(iou_scores))
        print("len_ok:",len_ok)

        return {
            "mean_iou": mean_iou,
        }
    else:
        return {
                "mean_iou": 0,
            }

  


# ===== Define the arguments =====
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, default=None)
args = parser.parse_args()
config_file=args.config
script_args, fed_args, peft_config, model_args, data_args = get_config(config_file)
training_args = get_training_args(script_args, script_args.learning_rate)
script_args = save_config(script_args, fed_args, model_args, data_args)

set_seed(script_args.seed)

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

with open("running_mixD.txt","a") as f:
    f.write(f"time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},current running file: {__file__}, config: {config_file} ,logdir: {logdir}\n")

logging.info("cuda: %s",torch.cuda.is_available())
device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
logging.info(f"Current memory allocated: {torch.cuda.memory_allocated(device) / (1024**3):.2f} GB")
logging.info("===============================================================")
logging.info("===============================================================")
logging.info("===============================================================")

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

    client_files = glob.glob(data_args.data_path + "train/client_*.json")
    print(f"Number of client files: {len(client_files)}")
    fed_args.num_clients = len(client_files)
    fed_args.sample_clients = len(client_files)
   

    local_datasets, global_test_datasets,sub_test_datasets,data_collator = make_supervised_data_module_mix_data_clients_iid(tokenizer=tokenizer, data_args=data_args,fed_args=fed_args,
                                                                                                                )

    sample_num_list = [len(local_datasets[i]) for i in range(fed_args.num_clients)]
    logging.info("sample_num_list: %s",sample_num_list) 

    # ===== Get model config =====

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    resume_from_checkpoint=training_args.resume_from_checkpoint
    select_keys=["eval_class_accuracy","eval_vqa_accuracy","eval_loss", "eval_mean_iou","eval_CIDEr","eval_ROUGE_L"]

    # ===== Define the global and local models =====
    share_params= get_selected_state_dict(model,avg_lora=fed_args.avg_lora,avg_pro=fed_args.avg_projector)
    global_dict = copy.deepcopy(share_params)
  
    upload_local_dict_list = [copy.deepcopy(global_dict) for _ in range(fed_args.num_clients)]




    if fed_args.fed_alg in ['fedprox','fedadagrad', 'fedyogi', 'fedadam',"fedavgm"]:
        for key in global_dict.keys():
            global_dict[key] = global_dict[key].to(device)
    
    local_params= get_selected_state_dict(model,avg_lora=not fed_args.avg_lora,avg_pro=not fed_args.avg_projector)
    save_local_list = [copy.deepcopy(local_params) for _ in range(fed_args.num_clients)]
    logging.info("lora_enable: %s,projercor_enable %s",fed_args.avg_lora,fed_args.avg_projector)
    
    logging.info("global_dict.keys(): %s", list(global_dict.keys())[0:10]+list(global_dict.keys())[-10:])
    logging.info("local_params.keys(): %s", local_params.keys())

  

    proxy_dict, opt_proxy_dict = get_proxy_dict(fed_args, global_dict)
    logging.info(f"Current memory allocated: {torch.cuda.max_memory_reserved(device) / (1024**3):.2f} GB")

    global_auxiliary, auxiliary_model_list, auxiliary_delta_dict = get_auxiliary_dict(fed_args, global_dict)
    logging.info(f"Current memory allocated: {torch.cuda.max_memory_reserved(device) / (1024**3):.2f} GB")


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
                compute_metrics=compute_acc_metrics,
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


            # if fed_args.test_local and round % 5 == 0 and round >=0 :
            #     print("local eval")
             
            #     for area_dataset in global_test_datasets:
            #         eval_results = trainer.evaluate(eval_dataset=area_dataset)
                   
            #         if "eval_accuracy" in eval_results:
            #                 task_name = "VQA"
            #         elif "eval_mean_iou" in eval_results:
            #             task_name = "Detection"
            #         else:
            #             task_name = "Caption"

            #         logging.info(task_name ,eval_results)
            #         for key in select_keys:
            #             if key in eval_results:
            #                 new_key = task_name+"_"+key
            #                 writer.add_scalar(f"Global_test/Average {new_key}", eval_results[key], round + 1)
            #         with open(os.path.join(script_args.output_dir, "eval_results.txt"), "a") as f:
            #             current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                       
            #             f.write(f"{current_time} - Round {round+1}:  client {client} {task_name} {eval_results}\n")
                                
            #         for key,value in eval_results.items():
            #             if key in select_keys:
            #                 new_key = task_name+"_"+key
            #                 test_metrics[client][new_key] = eval_results[key]


            torch.cuda.empty_cache()  
    
        # ===== summary loss =====
        avg_loss = np.mean([training_loss[i][-1] for i in range(fed_args.num_clients) if i in clients_this_round])
        writer.add_scalar('Train/Average Training Loss', avg_loss, round + 1)
        # if fed_args.test_local and round % 5 == 0 and round >=0:
        #     for task_name in ["VQA","Detection","Caption"]:
        #         for key in select_keys:
        #             new_key = task_name+"_"+key
        #             key_values = [test_metrics[client][new_key] for client in range(fed_args.num_clients) if new_key in test_metrics[client]]
        #             if key_values:
        #                 avg_key, std_key = compute_avg_and_std(key_values)
        #                 logging.info(f"Round {round+1}, Key: {new_key}, Avg: {avg_key}, Std: {std_key}")
        #                 writer.add_scalar(f"local_test_{new_key}/Avg", avg_key, round + 1)
        #                 writer.add_scalar(f"local_test_{new_key}/Std", std_key, round + 1)
        #                 with open(os.path.join(script_args.output_dir, "eval_results.txt"), "a") as f:
        #                     f.write(f"    summary - Round {round+1} : {new_key}, Avg: {avg_key}, Std: {std_key}\n")
  
        # ===== Server aggregates the local models =====
        for key in global_auxiliary.keys():
            global_auxiliary[key] = global_auxiliary[key].to(device)
        global_dict, global_auxiliary = global_aggregate(
            fed_args, global_dict, upload_local_dict_list, sample_num_list, \
            clients_this_round, round, proxy_dict=proxy_dict, \
            opt_proxy_dict=opt_proxy_dict, auxiliary_info=(global_auxiliary, auxiliary_delta_dict),use_cpu=False
        )
        # Log CUDA memory usage for important variables
        if  next(iter(global_dict.values())).device=="cuda":
            logging.info(f"CUDA memory usage for global_dict: {sum(param.numel() * param.element_size() for param in global_dict.values()) / (1024**3):.2f} GB")
        if  next(iter(upload_local_dict_list[0].values())).device=="cuda":
            logging.info(f"CUDA!!!")
            logging.info(f"CUDA memory usage for upload_local_dict_list: {sum(sum(param.numel() * param.element_size() for param in client_dict.values()) for client_dict in upload_local_dict_list) / (1024**3):.2f} GB")
        # logging.info(f"CUDA memory usage for auxiliary_model_list: {sum(sum(param.numel() * param.element_size() for param in model.values()) for model in auxiliary_model_list) / (1024**3):.2f} GB")
        # logging.info(f"CUDA memory usage for proxy_dict: {sum(param.numel() * param.element_size() for param in proxy_dict.values()) / (1024**3):.2f} GB")
        # logging.info(f"CUDA memory usage for global_auxiliary: {sum(param.numel() * param.element_size() for param in global_auxiliary.values()) / (1024**3):.2f} GB")
        
        set_selected_state_dict(model, global_dict, lora_enable=fed_args.avg_lora,projercor_enable=fed_args.avg_projector)
        torch.cuda.empty_cache()  

        # ===== Test the model =====
        logging.info(">> ========================================eval ==============") 
        logging.info(">> ========================================eval ==============") 
        logging.info(">> ========================================eval ==============") 

        for area_dataset in global_test_datasets:
          
            eval_results = trainer.evaluate(eval_dataset=area_dataset)
            logging.info(area_dataset.dataset_name ,eval_results)
            for key in select_keys:
                if key in eval_results:
                    writer.add_scalar(f"Global_test/Average {key}", eval_results[key], round + 1)
            with open(os.path.join(script_args.output_dir, "eval_results.txt"), "a") as f:
                current_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                if "eval_vqa_accuracy" in eval_results:
                    task_name = "VQA"
                elif "eval_class_accuracy" in eval_results:
                    task_name = "classification"
                elif "eval_mean_iou" in eval_results:
                    task_name = "Detection"
                else:
                    task_name = "Caption"
                f.write(f"{current_time} - Round {round+1}: {task_name} {eval_results}\n")


        if (round+1) % fed_args.save_model_freq == 0:
            logging.info(f"checkpoint-{round+1}")
            trainer.save_model(os.path.join(script_args.output_dir, f"checkpoint-{round+1}"))
        
        np.save(os.path.join(script_args.output_dir, "training_loss.npy"), np.array(training_loss))
        t_end=time.time()
        logging.info(f"round time: {(t_end-t_start)/3600:.2f} hours")

    with open("running_mixD.txt","a") as f:
        f.write(f"stop!!!! time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},current running file: {__file__}, logdir: {logdir}\n")
        f.write("======================================================\n")


except Exception as e:
    logging.error(e)
    logging.error("error")
    with open("running_mixD.txt","a") as f:
        f.write(f"stop!!!! time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')},current running file: {__file__}, logdir: {logdir}\n")
        f.write("======================================================\n")
    raise e