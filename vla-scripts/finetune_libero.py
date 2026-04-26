import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import draccus
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.distributed as dist
import tqdm
from ema_pytorch import EMA
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction_LIBERO
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransformLIBERO, RLDSBatchTransformLIBERO_withHis, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor
from prismatic.lifelong import (
    CLIPTextEmbedder,
    ODELoraConfig,
    SkillMemoryBank,
    configure_ode_lora_for_task,
    count_ode_lora_parameters,
    get_ode_lora_sparsity_loss,
    inject_ode_lora,
    load_instruction_samples,
    save_ode_lora_state,
)


os.environ["TOKENIZERS_PARALLELISM"] = "false"


from prismatic.models.policy.transformer_utils import MAPBlock

class ActionDecoder(torch.nn.Module):
    def __init__(self, window_size = 12, hidden_dim = 512):
        super().__init__()
        self.latent_action_pool = MAPBlock(n_latents = 1, vis_dim = 4096, embed_dim = hidden_dim, n_heads = hidden_dim // 64)
        self.visual_pool = MAPBlock(n_latents = 1, vis_dim = 4096, embed_dim = hidden_dim, n_heads = hidden_dim // 64)

        self.proj = nn.Sequential(
                                nn.Linear(hidden_dim, 7 * window_size),
                                nn.Tanh(),
                    )

    def forward(self, latent_action_tokens, visual_embed):
        visual_embed = self.visual_pool(visual_embed)
        latent_action_tokens = latent_action_tokens[:, -4:]
        action_token = self.latent_action_pool(latent_action_tokens, init_embed = visual_embed)

        action = self.proj(action_token)

        return action

class Wrapped_Model(torch.nn.Module):
    def __init__(self, vla, freeze_vla = False, window_size = 12):
        super().__init__()
        self.vla = vla
        self.window_size = window_size
        self.action_decoder = ActionDecoder(window_size=window_size)

        if freeze_vla:
            self.vla.requires_grad_(False)

    def forward(self, batch):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            vla_output = self.vla(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch["pixel_values"],
                labels=batch["labels"],
                output_hidden_states = True,
            )
        loss, loss_one_step, latent_action_tokens = self.action_decoder_forward(batch, vla_output)

        return vla_output, loss, loss_one_step, latent_action_tokens

    def action_decoder_forward(self, batch, vla_output):
        visual_embed = vla_output.hidden_states[-1][:, : self.vla.vision_backbone.featurizer.patch_embed.num_patches ].to(torch.float)
        latent_tokens = vla_output.hidden_states[-1][:, self.vla.vision_backbone.featurizer.patch_embed.num_patches : ]
        action_gt = batch["labels"].to(latent_tokens.device)
        mask = action_gt > 32000

        latent_action_tokens = []
        for idx, per_sample_latent_tokens in enumerate(latent_tokens):
            per_sample_latent_action_tokens = per_sample_latent_tokens[mask[idx], :]
            latent_action_tokens.append(per_sample_latent_action_tokens)
        latent_action_tokens = torch.stack(latent_action_tokens).to(torch.float)

        pred_action = self.action_decoder(latent_action_tokens, visual_embed).reshape(-1, self.window_size, 7)
        loss = torch.nn.functional.l1_loss(pred_action, batch['actions'], reduction='none')
        loss_one_step = loss[:,0].mean()
        loss = loss.mean()

        return loss, loss_one_step, latent_action_tokens


def _split_csv(value: str) -> Tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def _resolve_lifelong_instructions(cfg: "FinetuneConfig") -> List[str]:
    if cfg.lifelong_instruction_path is not None:
        instructions = load_instruction_samples(cfg.lifelong_instruction_path)
    elif cfg.lifelong_instruction_text is not None:
        instructions = load_instruction_samples(cfg.lifelong_instruction_text)
    else:
        instructions = []

    if not instructions:
        instructions = [cfg.dataset_name.replace("_", " ")]
    return instructions



@dataclass
class FinetuneConfig:

    vla_path: str = "./qwbu/univla-7b"
    lam_path: str = "./latent_action_model/lam-stage-2.ckpt"

    data_root_dir: Path = Path("./LIBERO/modified_libero_rlds")
    dataset_name: str = "libero_spatial_no_noops"
    run_root_dir: Path = Path("runs")
    adapter_tmp_dir: Path = Path("adapter-tmp")


    batch_size: int = 6
    max_steps: int = 30000000
    save_steps: int = 30000000
    learning_rate: float = 3.5e-4
    grad_accumulation_steps: int = 10
    image_aug: bool = True
    shuffle_buffer_size: int = 16000
    save_latest_checkpoint_only: bool = True



    codebook_size: int = 16
    lam_model_dim: int = 768
    lam_latent_dim: int = 128
    lam_patch_size: int = 14
    lam_enc_blocks: int = 12
    lam_dec_blocks: int = 12
    lam_num_heads: int = 12
    window_size: int = 12


    freeze_vla: bool = False
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    use_quantization: bool = False



    use_lifelong: bool = False
    lifelong_task_id: Optional[str] = None
    lifelong_task_category: str = "libero_manipulation"
    lifelong_instruction_path: Optional[str] = None
    lifelong_instruction_text: Optional[str] = None
    lifelong_memory_path: Path = Path("lifelong_memory/skill_memory.pt")
    lifelong_clip_model: str = "openai/clip-vit-base-patch32"
    lifelong_semantic_rank: int = 8
    lifelong_top_k: int = 2
    lifelong_expand_encoder: Optional[bool] = None
    lifelong_lora_rank: int = 32
    lifelong_lora_alpha: int = 16
    lifelong_expansion_rank: int = 8
    lifelong_decoder_nonzero: int = 32
    lifelong_spa_beta: float = 1e-4
    lifelong_tisi_temperature: float = 1.0
    lifelong_target_modules: str = "all-linear"
    lifelong_exclude_modules: str = "lm_head"


    wandb_project: str = "fientune-LIBERO"
    wandb_entity: str = "opendrivelab"
    run_id_note: Optional[str] = None



@draccus.wrap()
def finetune(cfg: FinetuneConfig) -> None:
    print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` on `{cfg.dataset_name}`")

    if cfg.use_lifelong and cfg.use_lora:
        print("Uni-SkillEvolver is enabled; disabling PEFT LoRA so ODE-LoRA owns the adapter path.")
        cfg.use_lora = False


    assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
    distributed_state = PartialState()
    torch.cuda.set_device(device_id := distributed_state.local_process_index)
    torch.cuda.empty_cache()


    exp_id = (
        f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
        f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
        f"+lr-{cfg.learning_rate}"
    )
    if cfg.use_lora:
        exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
    if cfg.use_lifelong:
        exp_id += (
            f"+uniskillevolver-r{cfg.lifelong_lora_rank}"
            f"+er{cfg.lifelong_expansion_rank}"
            f"+k{cfg.lifelong_top_k}"
        )
    if cfg.use_quantization:
        exp_id += "+q-4bit"
    if cfg.run_id_note is not None:
        exp_id += f"--{cfg.run_id_note}"
    if cfg.image_aug:
        exp_id += "--image_aug"

    exp_id += f'=w-LowLevelDecoder-ws-{cfg.window_size}'


    run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
    os.makedirs(run_dir, exist_ok=True)


    quantization_config = None
    if cfg.use_quantization:
        assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
        )


    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)


    processor = AutoProcessor.from_pretrained(cfg.vla_path, trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.vla_path,
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )



    if cfg.use_quantization:
        vla = prepare_model_for_kbit_training(vla)
    else:
        vla = vla.to(device_id)

    lifelong_memory = None
    lifelong_instructions = []
    lifelong_active_experts = []
    lifelong_task_id = cfg.lifelong_task_id or cfg.dataset_name
    lifelong_embedder = None


    if cfg.use_lora:
        lora_config = LoraConfig(
            r=cfg.lora_rank,
            lora_alpha=min(cfg.lora_rank, 16),
            lora_dropout=cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        vla = get_peft_model(vla, lora_config)
        vla.print_trainable_parameters()
    elif cfg.use_lifelong:
        lifelong_instructions = _resolve_lifelong_instructions(cfg)
        lifelong_memory = SkillMemoryBank.load(cfg.lifelong_memory_path)

        ode_config = ODELoraConfig(
            rank=cfg.lifelong_lora_rank,
            alpha=cfg.lifelong_lora_alpha,
            dropout=cfg.lora_dropout,
            expansion_rank=cfg.lifelong_expansion_rank,
            decoder_nonzero=cfg.lifelong_decoder_nonzero,
            coactivation_top_k=cfg.lifelong_top_k,
            target_modules=cfg.lifelong_target_modules,
            exclude_modules=_split_csv(cfg.lifelong_exclude_modules),
            tisi_enabled=True,
            tisi_temperature=cfg.lifelong_tisi_temperature,
            seed=0,
        )
        injection_report = inject_ode_lora(vla, ode_config)
        print("Injected Uni-SkillEvolver ODE-LoRA:", injection_report)

        retrieved_experts = []
        if len(lifelong_memory) > 0 and cfg.lifelong_top_k > 1:
            lifelong_embedder = CLIPTextEmbedder(
                model_name=cfg.lifelong_clip_model,
                device=torch.device(f"cuda:{device_id}"),
            )
            retrieved = lifelong_memory.retrieve(
                instruction=lifelong_instructions[0],
                embedder=lifelong_embedder,
                top_k=cfg.lifelong_top_k - 1,
            )
            for _, _, expert_group, _ in retrieved:
                for expert_id in expert_group:
                    if expert_id not in retrieved_experts and expert_id != lifelong_task_id:
                        retrieved_experts.append(expert_id)
                    if len(retrieved_experts) >= cfg.lifelong_top_k - 1:
                        break
                if len(retrieved_experts) >= cfg.lifelong_top_k - 1:
                    break

        if cfg.lifelong_expand_encoder is None:
            expand_encoder = len(lifelong_memory) > 0 and lifelong_memory.is_new_category(cfg.lifelong_task_category)
        else:
            expand_encoder = cfg.lifelong_expand_encoder

        lifelong_active_experts = configure_ode_lora_for_task(
            vla,
            task_id=lifelong_task_id,
            expand_encoder=expand_encoder,
            retrieved_expert_ids=retrieved_experts,
        )
        print("Configured Uni-SkillEvolver task:", {
            "task_id": lifelong_task_id,
            "category": cfg.lifelong_task_category,
            "expand_encoder": expand_encoder,
            "active_experts": lifelong_active_experts,
            **count_ode_lora_parameters(vla),
        })


    action_tokenizer = ActionTokenizer(processor.tokenizer)

    wrapped_model = Wrapped_Model(vla = vla, freeze_vla = cfg.freeze_vla, window_size=cfg.window_size).to(device_id)


    trainable_total_params = sum(p.numel() for p in wrapped_model.parameters() if p.requires_grad)
    print('Total Trainable Params: ', trainable_total_params)

    wrapped_model = DDP(wrapped_model, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)



    trainable_params = [param for param in wrapped_model.parameters() if param.requires_grad]
    optimizer = AdamW(trainable_params, lr=cfg.learning_rate, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = int(cfg.max_steps * 0.8), gamma=0.1)


    from latent_action_model.genie.modules.lam import ControllableDINOLatentActionModel

    latent_action_model = ControllableDINOLatentActionModel(
        in_dim=3,
        model_dim=cfg.lam_model_dim,
        latent_dim=cfg.lam_latent_dim,
        num_latents=cfg.codebook_size,
        patch_size=cfg.lam_patch_size,
        enc_blocks=cfg.lam_enc_blocks,
        dec_blocks=cfg.lam_dec_blocks,
        num_heads=cfg.lam_num_heads,
        dropout=0.,
    )

    lam_ckpt = torch.load(cfg.lam_path)['state_dict']
    new_ckpt = {}
    for key in lam_ckpt.keys():
        new_ckpt[key.replace("lam.", "")] = lam_ckpt[key]

    latent_action_model.load_state_dict(new_ckpt, strict=True)
    latent_action_model = latent_action_model.to(device_id).eval()

    batch_transform = RLDSBatchTransformLIBERO_withHis(
        latent_action_model,
        processor.tokenizer,
        image_transform=processor.image_processor.apply_transform,
        image_transform_lam=transforms.ToTensor(),
        prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
        window_size=cfg.window_size
    )


    vla_dataset = RLDSDataset(
        cfg.data_root_dir,
        cfg.dataset_name,
        batch_transform,
        resize_resolution=tuple(wrapped_model.module.vla.config.image_sizes),
        shuffle_buffer_size=cfg.shuffle_buffer_size,
        image_aug=cfg.image_aug,
        window_size=cfg.window_size + 1,
        training_phase='post-training',
    )


    if distributed_state.is_main_process:
        save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)


    collator = PaddedCollatorForActionPrediction_LIBERO(
        processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
    )
    dataloader = DataLoader(
        vla_dataset,
        batch_size=cfg.batch_size,
        sampler=None,
        collate_fn=collator,
        num_workers=0,
    )


    if distributed_state.is_main_process:
        wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")


    recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
    recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
    recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)


    with tqdm.tqdm(total=cfg.max_steps, leave=False) as progress:
        wrapped_model.train()
        optimizer.zero_grad()
        for batch_idx, batch in enumerate(dataloader):
            batch["input_ids"] = batch["input_ids"].to(device_id)
            batch["attention_mask"] = batch["attention_mask"].to(device_id)
            batch["labels"] = batch["labels"].to(device_id)
            batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16).to(device_id)
            batch['actions'] = batch['actions'].to(device_id)
            batch['latent_action_idx'] = batch['latent_action_idx'].to(device_id)


            output, act_loss, loss_one_step, latent_action_proj = wrapped_model(batch)
            loss = act_loss if cfg.freeze_vla else act_loss + output.loss
            tisi_sparsity_loss = None
            if cfg.use_lifelong:
                tisi_sparsity_loss = get_ode_lora_sparsity_loss(wrapped_model.module.vla)
                if tisi_sparsity_loss is not None:
                    loss = loss + cfg.lifelong_spa_beta * tisi_sparsity_loss


            normalized_loss = loss / cfg.grad_accumulation_steps
            torch.nn.utils.clip_grad_norm_(wrapped_model.parameters(), max_norm=1.)


            normalized_loss.backward()


            action_logits = output.logits[:, wrapped_model.module.vla.vision_backbone.featurizer.patch_embed.num_patches : -1]
            action_preds = action_logits.argmax(dim=2)
            action_gt = batch["labels"][:, 1:].to(action_preds.device)
            mask = action_gt > 32000


            correct_preds = (action_preds == action_gt) & mask
            action_accuracy = correct_preds.sum().float() / mask.sum().float()



            recent_losses.append(loss.item())
            recent_action_accuracies.append(action_accuracy.item())


            gradient_step_idx = batch_idx // cfg.grad_accumulation_steps




            smoothened_loss = sum(recent_losses) / len(recent_losses)
            smoothened_action_accuracy = sum(recent_action_accuracies) / len(recent_action_accuracies)


            if distributed_state.is_main_process and gradient_step_idx % 5 == 0:

                wandb.log(
                    {
                        "train_loss": smoothened_loss,
                        "latent_action_accuracy": smoothened_action_accuracy,
                        "action_loss": act_loss.item(),
                        "action_loss_1step": loss_one_step.item(),
                        "lr": optimizer.state_dict()['param_groups'][0]['lr'],
                        "tisi_sparsity_loss": 0.0
                        if tisi_sparsity_loss is None
                        else tisi_sparsity_loss.detach().float().item(),
                    },
                    step=gradient_step_idx,
                )


            if (batch_idx + 1) % cfg.grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                progress.update()


            if gradient_step_idx > 0 and gradient_step_idx % cfg.save_steps == 0:
                if distributed_state.is_main_process:
                    print(f"Saving Model Checkpoint for Step {gradient_step_idx}")


                    save_dir = adapter_dir if cfg.use_lora else run_dir


                    if cfg.use_lifelong:
                        processor.save_pretrained(run_dir)
                        save_ode_lora_state(
                            wrapped_model.module.vla,
                            run_dir / f"lifelong_adapter-{gradient_step_idx}.pt",
                            metadata={
                                "task_id": lifelong_task_id,
                                "task_category": cfg.lifelong_task_category,
                                "active_experts": lifelong_active_experts,
                                "dataset_name": cfg.dataset_name,
                            },
                        )
                        if lifelong_memory is not None:
                            if lifelong_embedder is None:
                                lifelong_embedder = CLIPTextEmbedder(
                                    model_name=cfg.lifelong_clip_model,
                                    device=torch.device(f"cuda:{device_id}"),
                                )
                            lifelong_memory.upsert(
                                task_id=lifelong_task_id,
                                category=cfg.lifelong_task_category,
                                instructions=lifelong_instructions,
                                expert_group=lifelong_active_experts,
                                head_name=cfg.dataset_name,
                                embedder=lifelong_embedder,
                                semantic_rank=cfg.lifelong_semantic_rank,
                            )
                            lifelong_memory.save(cfg.lifelong_memory_path)
                    elif not cfg.freeze_vla:
                        processor.save_pretrained(run_dir)
                        wrapped_model.module.vla.save_pretrained(save_dir)


                    torch.save(wrapped_model.module.action_decoder.state_dict(), str(run_dir) + f'/action_decoder-{gradient_step_idx}.pt')


                dist.barrier()



                if cfg.use_lora:
                    base_vla = AutoModelForVision2Seq.from_pretrained(
                        cfg.vla_path, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
                    )
                    merged_vla = PeftModel.from_pretrained(base_vla, adapter_dir)
                    merged_vla = merged_vla.merge_and_unload()
                    if distributed_state.is_main_process:
                        if cfg.save_latest_checkpoint_only:

                            merged_vla.save_pretrained(run_dir)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {run_dir}")
                        else:

                            checkpoint_dir = Path(str(run_dir) + f"--{gradient_step_idx}_chkpt")
                            os.makedirs(checkpoint_dir, exist_ok=True)


                            save_dataset_statistics(vla_dataset.dataset_statistics, checkpoint_dir)


                            processor.save_pretrained(checkpoint_dir)
                            merged_vla.save_pretrained(checkpoint_dir)

                            print(f"Saved Model Checkpoint for Step {gradient_step_idx} at: {checkpoint_dir}")


                dist.barrier()


            if gradient_step_idx == cfg.max_steps:
                print(f"Max step {cfg.max_steps} reached! Stopping training...")
                break


if __name__ == "__main__":
    finetune()
