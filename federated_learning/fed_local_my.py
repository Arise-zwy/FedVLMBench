import torch
import copy
import os
from transformers import TrainerCallback
from peft import get_peft_model_state_dict, set_peft_model_state_dict
from fairscale.optim.oss import OSS
import torch.nn as nn
from transformers import Trainer
from typing import List, Optional
from torch.utils.data import Sampler
# from transformers.trainer import (ALL_LAYERNORM_LAYERS, ShardedDDPOption,
#                                   get_parameter_names, has_length,
#                                   is_sagemaker_mp_enabled, logger)
from transformers.trainer import (ALL_LAYERNORM_LAYERS,
                                  get_parameter_names, has_length,nested_detach, 
                                  is_sagemaker_mp_enabled, logger)
from torch.cuda.amp import autocast, GradScaler

from transformers.utils import is_sagemaker_mp_enabled, is_apex_available

from transformers.trainer_pt_utils import (
    EvalLoopContainer)



if is_sagemaker_mp_enabled():
    from transformers.trainer_pt_utils import smp_forward_backward
if is_apex_available():
    from apex import amp
# formatting_prompts_func
def get_fed_local_vlm_sft_trainer(script_args, fed_args, model, tokenizer, training_args, local_dataset, eval_dataset, compute_metrics, data_collator, global_dict, local_auxiliary, global_auxiliary,preprocess_logits_for_metrics,resume_from_checkpoint):
    
    if fed_args.fed_alg == 'fedprox':
        trainer = VLMTrainerFedProx(
            model=model,
            tokenizer=tokenizer,
            args=training_args, 
            train_dataset=local_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            global_state=global_dict,
            prox_mu=fed_args.prox_mu,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,  
            resume_from_checkpoint=resume_from_checkpoint
        )
    elif fed_args.fed_alg == 'scaffold':
        trainer = VLMTrainerSCAFFOLD(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=local_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            global_state=global_dict,
            local_auxiliary=local_auxiliary,
            global_auxiliary=global_auxiliary,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,  
            resume_from_checkpoint=resume_from_checkpoint

        )
        trainer.add_callback(SCAFFOLD_Callback(trainer.correction, model))

    elif (fed_args.fed_alg in ['fedavg','fedavgm','fedyogi','fedadagrad','fedadam']) or (fed_args.fed_alg).startswith('local'):

    # elif (fed_args.fed_alg in ['fedavg']) or (fed_args.fed_alg).startswith('local'):
        trainer = VLMTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=local_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,  
            resume_from_checkpoint=resume_from_checkpoint
        )
    else:
        raise ValueError(f'Unsupported `fed_alg`: {fed_args.fed_alg}')
    return trainer

   
def get_local_vlm_sft_trainer(script_args, fed_args, model, tokenizer, training_args, local_dataset, eval_dataset, compute_metrics, data_collator,preprocess_logits_for_metrics,resume_from_checkpoint):
    
    if fed_args.fed_alg == 'fedprox':
        trainer = VLMTrainerFedProx(
            model=model,
            tokenizer=tokenizer,
            args=training_args, 
            train_dataset=local_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,  
            resume_from_checkpoint=resume_from_checkpoint,
        )
    elif fed_args.fed_alg == 'scaffold':
        trainer = VLMTrainerSCAFFOLD(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=local_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            resume_from_checkpoint=resume_from_checkpoint,
 
        )
        trainer.add_callback(SCAFFOLD_Callback(trainer.correction, model))
    elif (fed_args.fed_alg in ['fedavg']) or (fed_args.fed_alg).startswith('local'):
        trainer = VLMTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=local_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=compute_metrics,
            data_collator=data_collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
            resume_from_checkpoint=resume_from_checkpoint,
        )
    else:
        raise ValueError(f'Unsupported `fed_alg`: {fed_args.fed_alg}')
    return trainer

   

def split_to_even_chunks(indices, lengths, num_chunks):
    """
    Split a list of indices into `chunks` chunks of roughly equal lengths.
    """

    if len(indices) % num_chunks != 0:
        return [indices[i::num_chunks] for i in range(num_chunks)]

    num_indices_per_chunk = len(indices) // num_chunks

    chunks = [[] for _ in range(num_chunks)]
    chunks_lengths = [0 for _ in range(num_chunks)]
    for index in indices:
        shortest_chunk = chunks_lengths.index(min(chunks_lengths))
        chunks[shortest_chunk].append(index)
        chunks_lengths[shortest_chunk] += lengths[index]
        if len(chunks[shortest_chunk]) == num_indices_per_chunk:
            chunks_lengths[shortest_chunk] = float("inf")

    return chunks


def get_modality_length_grouped_indices(lengths, batch_size, world_size, generator=None):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    assert all(l != 0 for l in lengths), "Should not have zero length."
    if all(l > 0 for l in lengths) or all(l < 0 for l in lengths):
        # all samples are in the same modality
        return get_length_grouped_indices(lengths, batch_size, world_size, generator=generator)
    mm_indices, mm_lengths = zip(*[(i, l) for i, l in enumerate(lengths) if l > 0])
    lang_indices, lang_lengths = zip(*[(i, -l) for i, l in enumerate(lengths) if l < 0])

    # assert len(mm_indices) > 0, "Should have at least one multimodal sample."
    # assert len(lang_indices) > 0, "Should have at least one language sample."

    mm_shuffle = [mm_indices[i] for i in get_length_grouped_indices(mm_lengths, batch_size, world_size, generator=None)]
    lang_shuffle = [lang_indices[i] for i in get_length_grouped_indices(lang_lengths, batch_size, world_size, generator=None)]
    megabatch_size = world_size * batch_size
    mm_megabatches = [mm_shuffle[i : i + megabatch_size] for i in range(0, len(mm_shuffle), megabatch_size)]
    lang_megabatches = [lang_shuffle[i : i + megabatch_size] for i in range(0, len(lang_shuffle), megabatch_size)]

    last_mm = mm_megabatches[-1]
    last_lang = lang_megabatches[-1]
    additional_batch = last_mm + last_lang
    megabatches = mm_megabatches[:-1] + lang_megabatches[:-1]
    megabatch_indices = torch.randperm(len(megabatches), generator=generator)
    megabatches = [megabatches[i] for i in megabatch_indices]

    if len(additional_batch) >= megabatch_size:
        megabatches = [additional_batch[:megabatch_size]] + megabatches
        additional_batch = additional_batch[megabatch_size:]

    if len(additional_batch) > 0:
        megabatches.append(additional_batch)

    return [i for megabatch in megabatches for i in megabatch]


def get_length_grouped_indices(lengths, batch_size, world_size, generator=None, merge=True):
    # We need to use torch for the random part as a distributed sampler will set the random seed for torch.
    indices = torch.randperm(len(lengths), generator=generator)
    megabatch_size = world_size * batch_size
    megabatches = [indices[i : i + megabatch_size].tolist() for i in range(0, len(lengths), megabatch_size)]
    megabatches = [sorted(megabatch, key=lambda i: lengths[i], reverse=True) for megabatch in megabatches]
    megabatches = [split_to_even_chunks(megabatch, lengths, world_size) for megabatch in megabatches]

    return [i for megabatch in megabatches for batch in megabatch for i in batch]


class LengthGroupedSampler(Sampler):
    r"""
    Sampler that samples indices in a way that groups together features of the dataset of roughly the same length while
    keeping a bit of randomness.
    """

    def __init__(
        self,
        batch_size: int,
        world_size: int,
        lengths: Optional[List[int]] = None,
        generator=None,
        group_by_modality: bool = False,
    ):
        if lengths is None:
            raise ValueError("Lengths must be provided.")

        self.batch_size = batch_size
        self.world_size = world_size
        self.lengths = lengths
        self.generator = generator
        self.group_by_modality = group_by_modality

    def __len__(self):
        return len(self.lengths)

    def __iter__(self):
        if self.group_by_modality:
            indices = get_modality_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        else:
            indices = get_length_grouped_indices(self.lengths, self.batch_size, self.world_size, generator=self.generator)
        return iter(indices)

  


class VLMTester(Trainer):
    def __init__(self, model, tokenizer,args, eval_dataset,compute_metrics,data_collator,preprocess_logits_for_metrics):
        super(VLMTester,self).__init__( model=model,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            args=args,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics) 
      


    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys,
    ) :
       
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"]+["labels"])
                else:
                    logits = outputs[1:]


        labels= outputs["labels"]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    


class VLMTrainer(Trainer):
    def __init__(self, model, tokenizer,args,train_dataset, eval_dataset,compute_metrics,data_collator,preprocess_logits_for_metrics,resume_from_checkpoint=None):
        super(VLMTrainer,self).__init__( model=model,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics) 
        self.scaler = GradScaler()
        self.do_grad_scaling=False
        self.use_apex=False
        if resume_from_checkpoint:
            self._load_checkpoint(resume_from_checkpoint)
   

    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # if self.args.group_by_modality_length:
        if True:
            lengths = self.train_dataset.modality_lengths
            # print("self.train_dataset.modality_lengths",self.train_dataset.modality_lengths)
            return LengthGroupedSampler(
                # TODO: seems that we should not have gradient_accumulation_steps
                # self.args.train_batch_size * self.args.gradient_accumulation_steps,
                self.args.train_batch_size,
                world_size=self.args.world_size,
                lengths=lengths,
                group_by_modality=True,
            )
        else:
            return super()._get_train_sampler()

    def create_optimizer(self):
        """
        Setup the optimizer.

        We provide a reasonable default that works well. If you want to use something else, you can pass a tuple in the
        Trainer's init through `optimizers`, or subclass and override this method in a subclass.
        """
        if is_sagemaker_mp_enabled():
            return super().create_optimizer()
      
        opt_model = self.model

        if self.optimizer is None:
            decay_parameters = get_parameter_names(
                opt_model, ALL_LAYERNORM_LAYERS)
            decay_parameters = [
                name for name in decay_parameters if "bias" not in name]
            # here we can set the vision tower to be not trainable
            train_vision=False
            if train_vision==False:
                unused_parameters = [
                    name for name, _ in opt_model.named_parameters() if "vision_tower" in name and "layers" not in name
                ]
            else:
                unused_parameters = []
            
            # here we can set whether to make learning rate of mm_projector different
            if self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0:
                projector_parameters = [
                    name for name, _ in opt_model.named_parameters() if "mm_projector" in name]
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n not in projector_parameters and n not in unused_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n not in projector_parameters and n not in unused_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and n in projector_parameters and n not in unused_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                        "lr": self.args.mm_projector_lr,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and n in projector_parameters and n not in unused_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                        "lr": self.args.mm_projector_lr,
                    },
                ]
            else:
                optimizer_grouped_parameters = [
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": self.args.weight_decay,
                    },
                    {
                        "params": [
                            p for n, p in opt_model.named_parameters() if (n not in decay_parameters and p.requires_grad)
                        ],
                        "weight_decay": 0.0,
                    },
                ]

            optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                self.args)
            
       
            self.optimizer = optimizer_cls(
                optimizer_grouped_parameters, **optimizer_kwargs)
            if optimizer_cls.__name__ == "Adam8bit":
                import bitsandbytes

                manager = bitsandbytes.optim.GlobalOptimManager.get_instance()

                skipped = 0
                for module in opt_model.modules():
                    if isinstance(module, nn.Embedding):
                        skipped += sum({p.data_ptr(): p.numel()
                                        for p in module.parameters()}.values())
                        logger.info(
                            f"skipped {module}: {skipped/2**20}M params")
                        manager.register_module_override(
                            module, "weight", {"optim_bits": 32})
                        logger.debug(
                            f"bitsandbytes: will optimize {module} in fp32")
                logger.info(f"skipped: {skipped/2**20}M params")

        return self.optimizer

    def training_step(self, model, inputs):
        """
        Perform a training step on a batch of inputs.

        Subclass and override to inject custom behavior.

        Args:
            model (`nn.Module`):
                The model to train.
            inputs (`Dict[str, Union[torch.Tensor, Any]]`):
                The inputs and targets of the model.

                The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
                argument `labels`. Check your model's documentation for all accepted arguments.

        Return:
            `torch.Tensor`: The tensor with training loss on this batch.
        """

        model.train()

        inputs = self._prepare_inputs(inputs)

     

        # if is_sagemaker_mp_enabled():
        #     loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
        #     return loss_mb.reduce_mean().detach().to(self.args.device)
        is_detection=False
        if is_detection:
            with autocast(enabled=True):
                with self.compute_loss_context_manager():
                    loss = self.compute_loss_detection(model, inputs,return_outputs=False)
        
        else:
            with autocast(enabled=True):
                with self.compute_loss_context_manager():
                    loss = self.compute_loss(model, inputs,return_outputs=False)

        if self.args.n_gpu > 1:
            loss = loss.mean()

        if self.do_grad_scaling:
            self.scaler.scale(loss).backward()
        elif self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
        
            self.accelerator.backward(loss)

        # if self.use_apex:
        #     with amp.scale_loss(loss, self.optimizer) as scaled_loss:
        #         scaled_loss.backward()
        # else:
        #     self.accelerator.backward(loss)


        return loss.detach() / self.args.gradient_accumulation_steps

    def prediction_step(
        self,
        model,
        inputs,
        prediction_loss_only,
        ignore_keys,
    ) :
       
        has_labels = False if len(self.label_names) == 0 else all(inputs.get(k) is not None for k in self.label_names)
        # For CLIP-like models capable of returning loss values.
        # If `return_loss` is not specified or being `None` in `inputs`, we check if the default value of `return_loss`
        # is `True` in `model.forward`.
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = True if len(self.label_names) == 0 and return_loss else False

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(self.model.config, "keys_to_ignore_at_inference", [])
            else:
                ignore_keys = []

        # labels may be popped when computing the loss (label smoothing for instance) so we grab them first.
        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name) for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if has_labels or loss_without_labels:
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                loss = loss.mean().detach()

                if isinstance(outputs, dict):
                    logits = tuple(v for k, v in outputs.items() if k not in ignore_keys + ["loss"]+["labels"])
                else:
                    logits = outputs[1:]


        labels= outputs["labels"]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)

    
    def compute_loss_detection(self, model, inputs, return_outputs=False):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
     
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:            
            loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

class VLMTrainerFedProx(VLMTrainer):
    def __init__(self, global_state, prox_mu, **kwargs):
        super(VLMTrainerFedProx, self).__init__(**kwargs)
        self.global_state = global_state
        self.mu = prox_mu

    def compute_loss(self, model, inputs, return_outputs=False):

        return_values = super(VLMTrainerFedProx, self).compute_loss(model, inputs, return_outputs=return_outputs)

        if return_outputs:
            loss, outputs = return_values
        else:
            loss = return_values

        # Apply FedProx Loss
        for name, param in model.named_parameters():
            name = name.replace(".default", "")     # TODO: May need changes. to accord with peft
            # only trainable parameters
            if not param.requires_grad:
                continue
            else:
                loss += self.mu / 2 * torch.norm(param - self.global_state[name].detach()) ** 2
                # loss += self.mu / 2 * torch.norm(param - self.global_state[name]) ** 2

        return (loss, outputs) if return_outputs else loss

class VLMTrainerSCAFFOLD(VLMTrainer):
    def __init__(self, global_state, local_auxiliary, global_auxiliary, **kwargs):
        super(VLMTrainerSCAFFOLD, self).__init__(**kwargs)
        self.global_state = global_state
        self.local_auxiliary = local_auxiliary
        self.global_auxiliary = global_auxiliary
        self.correction = copy.deepcopy(local_auxiliary)

        for name in self.correction.keys():
            self.correction[name] = self.global_auxiliary[name] - self.local_auxiliary[name]
    

   
    def get_auxiliary_param(self):
        auxiliary_new_para = copy.deepcopy(self.local_auxiliary)
        auxiliary_delta_para = copy.deepcopy(self.local_auxiliary)
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                else:
                    name = name.replace(".default", "")
                    # 0315 make param to cpu 
                    param=param.cpu()
                    self.global_state[name]=self.global_state[name].cpu()

                    auxiliary_new_para[name] = (self.global_state[name] - param) / (self.args.max_steps * self.args.learning_rate) - self.correction[name]
                    auxiliary_delta_para[name] = auxiliary_new_para[name] - self.local_auxiliary[name]
        return auxiliary_new_para, auxiliary_delta_para


# need to rewrite the callback
class SCAFFOLD_Callback(TrainerCallback):
    def __init__(self, correction, model):
        super(SCAFFOLD_Callback, self).__init__()
        self.correction = correction
        self.model = model
    def on_step_end(self, args, state, control, **kwargs):
        model_para = copy.deepcopy(get_peft_model_state_dict(self.model))

        for name in model_para.keys():
            model_para[name]=model_para[name].cpu()
            model_para[name] -= args.learning_rate * self.correction[name]
        set_peft_model_state_dict(self.model, model_para)


# class SFTTrainerFedProx(SFTTrainer):
#     def __init__(self, global_state, prox_mu, **kwargs):
#         super(SFTTrainerFedProx, self).__init__(**kwargs)
#         self.global_state = global_state
#         self.mu = prox_mu
    
#     def compute_loss(self, model, inputs, return_outputs=False):

#         return_values = super(SFTTrainerFedProx, self).compute_loss(model, inputs, return_outputs=return_outputs)

#         if return_outputs:
#             loss, outputs = return_values
#         else:
#             loss = return_values

#         # Apply FedProx Loss
#         for name, param in model.named_parameters():
#             name = name.replace(".default", "")     # TODO: May need changes. to accord with peft
#             # only trainable parameters
#             if not param.requires_grad:
#                 continue
#             else:
#                 loss += self.mu / 2 * torch.norm(param - self.global_state[name]) ** 2

#         return (loss, outputs) if return_outputs else loss


# class SFTTrainerSCAFFOLD(SFTTrainer):
#     def __init__(self, global_state, local_auxiliary, global_auxiliary, **kwargs):
#         super(SFTTrainerSCAFFOLD, self).__init__(**kwargs)
#         self.global_state = global_state
#         self.local_auxiliary = local_auxiliary
#         self.global_auxiliary = global_auxiliary
#         self.correction = copy.deepcopy(local_auxiliary)

#         for name in self.correction.keys():
#             self.correction[name] = self.global_auxiliary[name] - self.local_auxiliary[name]
    
#     def get_auxiliary_param(self):
#         auxiliary_new_para = copy.deepcopy(self.local_auxiliary)
#         auxiliary_delta_para = copy.deepcopy(self.local_auxiliary)
#         with torch.no_grad():
#             for name, param in self.model.named_parameters():
#                 if not param.requires_grad:
#                     continue
#                 else:
#                     name = name.replace(".default", "")
#                     auxiliary_new_para[name] = (self.global_state[name] - param) / (self.args.max_steps * self.args.learning_rate) - self.correction[name]
#                     auxiliary_delta_para[name] = auxiliary_new_para[name] - self.local_auxiliary[name]
#         return auxiliary_new_para, auxiliary_delta_para

# class SCAFFOLD_Callback(TrainerCallback):
#     def __init__(self, correction, model):
#         super(SCAFFOLD_Callback, self).__init__()
#         self.correction = correction
#         self.model = model
#     def on_step_end(self, args, state, control, **kwargs):
#         model_para = copy.deepcopy(get_peft_model_state_dict(self.model))
#         for name in model_para.keys():
#             model_para[name] -= args.learning_rate * self.correction[name]
#         set_peft_model_state_dict(self.model, model_para)



 