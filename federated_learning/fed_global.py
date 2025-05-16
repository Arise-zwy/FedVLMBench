import random
import torch

def get_clients_this_round(fed_args, round):
    if (fed_args.fed_alg).startswith('local'):
        clients_this_round = [int((fed_args.fed_alg)[-1])]
    else:
        if fed_args.num_clients < fed_args.sample_clients:
            clients_this_round = list(range(fed_args.num_clients))
        else:
            random.seed(round)
            clients_this_round = sorted(random.sample(range(fed_args.num_clients), fed_args.sample_clients))
    return clients_this_round


def global_aggregate(fed_args, global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, proxy_dict=None, opt_proxy_dict=None, auxiliary_info=None,use_cpu=False):
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    global_auxiliary = None
    if use_cpu==False:
        if fed_args.fed_alg == 'scaffold':
            for key in global_dict.keys():
                global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
            global_auxiliary, auxiliary_delta_dict = auxiliary_info
            for key in global_auxiliary.keys():
                delta_auxiliary = sum([auxiliary_delta_dict[client][key] for client in clients_this_round]) 
                global_auxiliary[key] += delta_auxiliary / fed_args.num_clients
        
        elif fed_args.fed_alg == 'fedavgm':
            # Momentum-based FedAvg
            for key in global_dict.keys():
                delta_w = sum([(local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round for client in clients_this_round])
                proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                global_dict[key] = global_dict[key] + proxy_dict[key]

        elif fed_args.fed_alg == 'fedadagrad':
            for key, param in opt_proxy_dict.items():
                delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
                # In paper 'adaptive federated optimization', momentum is not used
                proxy_dict[key] = delta_w
                opt_proxy_dict[key] = param + torch.square(proxy_dict[key])
                global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

        elif fed_args.fed_alg == 'fedyogi':
            for key, param in opt_proxy_dict.items():
                delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
                proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                delta_square = torch.square(proxy_dict[key])
                opt_proxy_dict[key] = param - (1-fed_args.fedopt_beta2)*delta_square*torch.sign(param - delta_square)
                global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

        elif fed_args.fed_alg == 'fedadam':
            for key, param in opt_proxy_dict.items():
                delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
                proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                opt_proxy_dict[key] = fed_args.fedopt_beta2*param + (1-fed_args.fedopt_beta2)*torch.square(proxy_dict[key])
                global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

        else:   # Normal dataset-size-based aggregation 
            for key in global_dict.keys():
                global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
    else:
        device = torch.device("cuda")
        if fed_args.fed_alg == 'scaffold':
            for key in global_dict.keys():
                global_dict[key] = sum([local_dict_list[client][key].cpu() * sample_num_list[client] / sample_this_round for client in clients_this_round]).to(device)
            global_auxiliary, auxiliary_delta_dict = auxiliary_info
            for key in global_auxiliary.keys():
                delta_auxiliary = sum([auxiliary_delta_dict[client][key].cpu() for client in clients_this_round])
                global_auxiliary[key] += (delta_auxiliary / fed_args.num_clients).to(device)
        
        elif fed_args.fed_alg == 'fedavgm':
            for key in global_dict.keys():
                delta_w = sum([(local_dict_list[client][key].cpu() - global_dict[key].cpu()) * sample_num_list[client] / sample_this_round for client in clients_this_round])
                proxy_dict[key] = (fed_args.fedopt_beta1 * proxy_dict[key].cpu() + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w).to(device)
                # print(proxy_dict[key].device,global_dict[key].device)
                global_dict[key] = (global_dict[key].cpu() + proxy_dict[key].cpu()).to(device)

        elif fed_args.fed_alg == 'fedadagrad':
            for key, param in opt_proxy_dict.items():
                delta_w = sum([(local_dict_list[client][key].cpu() - global_dict[key].cpu()) for client in clients_this_round]) / len(clients_this_round)
                proxy_dict[key] = delta_w.to(device)
                opt_proxy_dict[key] = (param.cpu() + torch.square(proxy_dict[key].cpu())).to(device)
                global_dict[key] += (fed_args.fedopt_eta * torch.div(proxy_dict[key].cpu(), torch.sqrt(opt_proxy_dict[key].cpu()) + fed_args.fedopt_tau)).to(device)

        elif fed_args.fed_alg == 'fedyogi':
            for key, param in opt_proxy_dict.items():
                delta_w = sum([(local_dict_list[client][key].cpu() - global_dict[key].cpu()) for client in clients_this_round]) / len(clients_this_round)
                proxy_dict[key] = (fed_args.fedopt_beta1 * proxy_dict[key].cpu() + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w).to(device)
                delta_square = torch.square(proxy_dict[key].cpu())
                opt_proxy_dict[key] = (param.cpu() - (1 - fed_args.fedopt_beta2) * delta_square * torch.sign(param.cpu() - delta_square)).to(device)
                global_dict[key] += (fed_args.fedopt_eta * torch.div(proxy_dict[key].cpu(), torch.sqrt(opt_proxy_dict[key].cpu()) + fed_args.fedopt_tau)).to(device)

        elif fed_args.fed_alg == 'fedadam':
            for key, param in opt_proxy_dict.items():
                delta_w = sum([(local_dict_list[client][key].cpu() - global_dict[key].cpu()) for client in clients_this_round]) / len(clients_this_round)
                proxy_dict[key] = (fed_args.fedopt_beta1 * proxy_dict[key].cpu() + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w).to(device)
                opt_proxy_dict[key] = (fed_args.fedopt_beta2 * param.cpu() + (1 - fed_args.fedopt_beta2) * torch.square(proxy_dict[key].cpu())).to(device)
                print("global_dict[key]",global_dict[key].device)
                print("proxy_dict[key]",proxy_dict[key].device)
                print("opt_proxy_dict[key]",opt_proxy_dict[key].device)
                global_dict[key] += (fed_args.fedopt_eta * torch.div(proxy_dict[key].cpu(), torch.sqrt(opt_proxy_dict[key].cpu()) + fed_args.fedopt_tau)).to(device)

        else:   # Normal dataset-size-based aggregation 
            for key in global_dict.keys():
                global_dict[key] = sum([local_dict_list[client][key].cpu() * sample_num_list[client] / sample_this_round for client in clients_this_round]).to(device)
        
    return global_dict, global_auxiliary

def global_aggregate_showo(fed_args, global_dict, local_dict_list, sample_num_list, clients_this_round, round_idx, proxy_dict=None, opt_proxy_dict=None, auxiliary_info=None,use_cpu=False):
    sample_this_round = sum([sample_num_list[client] for client in clients_this_round])
    global_auxiliary = None
    if use_cpu==False:
        if fed_args.fed_alg == 'scaffold':
            for key in global_dict.keys():
                global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
            global_auxiliary, auxiliary_delta_dict = auxiliary_info
            for key in global_auxiliary.keys():
                delta_auxiliary = sum([auxiliary_delta_dict[client][key] for client in clients_this_round]) 
                global_auxiliary[key] += delta_auxiliary / fed_args.num_clients
        
        elif fed_args.fed_alg == 'fedavgm':
            # Momentum-based FedAvg
            for key in global_dict.keys():
                delta_w = sum([(local_dict_list[client][key] - global_dict[key]) * sample_num_list[client] / sample_this_round for client in clients_this_round])
                proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                global_dict[key] = global_dict[key] + proxy_dict[key]

        elif fed_args.fed_alg == 'fedadagrad':
            for key, param in opt_proxy_dict.items():
                delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
                # In paper 'adaptive federated optimization', momentum is not used
                proxy_dict[key] = delta_w
                opt_proxy_dict[key] = param + torch.square(proxy_dict[key])
                global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

        elif fed_args.fed_alg == 'fedyogi':
            for key, param in opt_proxy_dict.items():
                delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
                proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                delta_square = torch.square(proxy_dict[key])
                opt_proxy_dict[key] = param - (1-fed_args.fedopt_beta2)*delta_square*torch.sign(param - delta_square)
                global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

        elif fed_args.fed_alg == 'fedadam':
            for key, param in opt_proxy_dict.items():
                delta_w = sum([(local_dict_list[client][key] - global_dict[key]) for client in clients_this_round]) / len(clients_this_round)
                proxy_dict[key] = fed_args.fedopt_beta1 * proxy_dict[key] + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w
                opt_proxy_dict[key] = fed_args.fedopt_beta2*param + (1-fed_args.fedopt_beta2)*torch.square(proxy_dict[key])
                global_dict[key] += fed_args.fedopt_eta * torch.div(proxy_dict[key], torch.sqrt(opt_proxy_dict[key])+fed_args.fedopt_tau)

        else:   # Normal dataset-size-based aggregation 
            for key in global_dict.keys():
                global_dict[key] = sum([local_dict_list[client][key] * sample_num_list[client] / sample_this_round for client in clients_this_round])
    else:
        device = torch.device("cpu")
        if fed_args.fed_alg == 'scaffold':
            for key in global_dict.keys():
                global_dict[key] = sum([local_dict_list[client][key].cpu() * sample_num_list[client] / sample_this_round for client in clients_this_round]).to(device)
            global_auxiliary, auxiliary_delta_dict = auxiliary_info
            for key in global_auxiliary.keys():
                delta_auxiliary = sum([auxiliary_delta_dict[client][key].cpu() for client in clients_this_round])
                global_auxiliary[key] += (delta_auxiliary / fed_args.num_clients).to(device)
        
        elif fed_args.fed_alg == 'fedavgm':
            for key in global_dict.keys():
                delta_w = sum([(local_dict_list[client][key].cpu() - global_dict[key].cpu()) * sample_num_list[client] / sample_this_round for client in clients_this_round])
                proxy_dict[key] = (fed_args.fedopt_beta1 * proxy_dict[key].cpu() + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w).to(device)
                # print(proxy_dict[key].device,global_dict[key].device)
                global_dict[key] = (global_dict[key].cpu() + proxy_dict[key].cpu()).to(device)

        elif fed_args.fed_alg == 'fedadagrad':
            for key, param in opt_proxy_dict.items():
                delta_w = sum([(local_dict_list[client][key].cpu() - global_dict[key].cpu()) for client in clients_this_round]) / len(clients_this_round)
                proxy_dict[key] = delta_w.to(device)
                opt_proxy_dict[key] = (param.cpu() + torch.square(proxy_dict[key].cpu())).to(device)
                global_dict[key] += (fed_args.fedopt_eta * torch.div(proxy_dict[key].cpu(), torch.sqrt(opt_proxy_dict[key].cpu()) + fed_args.fedopt_tau)).to(device)

        elif fed_args.fed_alg == 'fedyogi':
            for key, param in opt_proxy_dict.items():
                delta_w = sum([(local_dict_list[client][key].cpu() - global_dict[key].cpu()) for client in clients_this_round]) / len(clients_this_round)
                proxy_dict[key] = (fed_args.fedopt_beta1 * proxy_dict[key].cpu() + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w).to(device)
                delta_square = torch.square(proxy_dict[key].cpu())
                opt_proxy_dict[key] = (param.cpu() - (1 - fed_args.fedopt_beta2) * delta_square * torch.sign(param.cpu() - delta_square)).to(device)
                global_dict[key] += (fed_args.fedopt_eta * torch.div(proxy_dict[key].cpu(), torch.sqrt(opt_proxy_dict[key].cpu()) + fed_args.fedopt_tau)).to(device)

        elif fed_args.fed_alg == 'fedadam':
            for key, param in opt_proxy_dict.items():
                delta_w = sum([(local_dict_list[client][key].cpu() - global_dict[key].cpu()) for client in clients_this_round]) / len(clients_this_round)
                proxy_dict[key] = (fed_args.fedopt_beta1 * proxy_dict[key].cpu() + (1 - fed_args.fedopt_beta1) * delta_w if round_idx > 0 else delta_w).to(device)
                opt_proxy_dict[key] = (fed_args.fedopt_beta2 * param.cpu() + (1 - fed_args.fedopt_beta2) * torch.square(proxy_dict[key].cpu())).to(device)
                global_dict[key] += (fed_args.fedopt_eta * torch.div(proxy_dict[key].cpu(), torch.sqrt(opt_proxy_dict[key].cpu()) + fed_args.fedopt_tau)).to(device)

        else:   # Normal dataset-size-based aggregation 
            for key in global_dict.keys():
                global_dict[key] = sum([local_dict_list[client][key].cpu() * sample_num_list[client] / sample_this_round for client in clients_this_round]).to(device)
        
    return global_dict, global_auxiliary