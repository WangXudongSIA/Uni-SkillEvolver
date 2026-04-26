export CUDA_VISIBLE_DEVICES=6,7

# Task1
torchrun --standalone --nnodes 1 --nproc-per-node 2 \
  vla-scripts/finetune_realworld.py \
  --use_lifelong True \
  --use_lora False \
  --vla_path ./qwbu/univla-7b \
  --lam_path ./latent_action_model/lam-stage-2.ckpt \
  --data_root_dir ./realworld \
  --dataset_name realworld \
  --lifelong_task_id task01_realworld \
  --lifelong_task_category realworld_manipulation \
  --lifelong_instruction_path lifelong_data/instructions/task01_realworld.txt \
  --lifelong_memory_path lifelong_memory/skill_memory.pt \
  --lifelong_top_k 3 \
  --lifelong_lora_rank 8 \
  --lifelong_lora_alpha 16 \
  --lifelong_expansion_rank 4 \
  --lifelong_decoder_nonzero 32 \
  --lifelong_spa_beta 1e-4 \
  --batch_size 8 \
  --grad_accumulation_steps 10 \
  --max_steps 30000 \
  --save_steps 30000 \
  --run_root_dir logs/lifelong/task01_realworld

# Task2
torchrun --standalone --nnodes 1 --nproc-per-node 2 \
  vla-scripts/finetune_r2r.py \
  --use_lifelong True \
  --use_lora False \
  --vla_path ./qwbu/univla-7b \
  --lam_path ./latent_action_model/lam-stage-2.ckpt \
  --data_root_dir ./r2r \
  --dataset_name r2r \
  --lifelong_task_id task02_r2r \
  --lifelong_task_category r2r_VLN \
  --lifelong_instruction_path lifelong_data/instructions/task02_r2r.txt \
  --lifelong_memory_path lifelong_memory/skill_memory.pt \
  --lifelong_top_k 3 \
  --lifelong_lora_rank 8 \
  --lifelong_lora_alpha 16 \
  --lifelong_expansion_rank 4 \
  --lifelong_decoder_nonzero 32 \
  --lifelong_spa_beta 1e-4 \
  --batch_size 8 \
  --grad_accumulation_steps 10 \
  --max_steps 30000 \
  --save_steps 30000 \
  --run_root_dir logs/lifelong/task02_r2r

# Task3
torchrun --standalone --nnodes 1 --nproc-per-node 2 \
  vla-scripts/finetune_realworld.py \
  --use_lifelong True \
  --use_lora False \
  --vla_path ./qwbu/univla-7b \
  --lam_path ./latent_action_model/lam-stage-2.ckpt \
  --data_root_dir ./realworld \
  --dataset_name realworld \
  --lifelong_task_id task03_realworld \
  --lifelong_task_category realworld_manipulation \
  --lifelong_instruction_path lifelong_data/instructions/task03_realworld.txt \
  --lifelong_memory_path lifelong_memory/skill_memory.pt \
  --lifelong_top_k 3 \
  --lifelong_lora_rank 8 \
  --lifelong_lora_alpha 16 \
  --lifelong_expansion_rank 4 \
  --lifelong_decoder_nonzero 32 \
  --lifelong_spa_beta 1e-4 \
  --batch_size 8 \
  --grad_accumulation_steps 10 \
  --max_steps 30000 \
  --save_steps 30000 \
  --run_root_dir logs/lifelong/task03_realworld

# Task4
torchrun --standalone --nnodes 1 --nproc-per-node 2 \
  vla-scripts/finetune_libero.py \
  --use_lifelong True \
  --use_lora False \
  --vla_path ./qwbu/univla-7b \
  --lam_path ./latent_action_model/lam-stage-2.ckpt \
  --data_root_dir ./LIBERO/modified_libero_rlds \
  --dataset_name libero_object_no_noops \
  --lifelong_task_id task04_libero_object \
  --lifelong_task_category libero_manipulation \
  --lifelong_instruction_path lifelong_data/instructions/task04_libero_object.txt \
  --lifelong_memory_path lifelong_memory/skill_memory.pt \
  --lifelong_top_k 3 \
  --lifelong_lora_rank 8 \
  --lifelong_lora_alpha 16 \
  --lifelong_expansion_rank 4 \
  --lifelong_decoder_nonzero 32 \
  --lifelong_spa_beta 1e-4 \
  --batch_size 8 \
  --grad_accumulation_steps 10 \
  --max_steps 30000 \
  --save_steps 30000 \
  --run_root_dir logs/lifelong/task04_libero_object

# Task5
torchrun --standalone --nnodes 1 --nproc-per-node 2 \
  vla-scripts/finetune_r2r.py \
  --use_lifelong True \
  --use_lora False \
  --vla_path ./qwbu/univla-7b \
  --lam_path ./latent_action_model/lam-stage-2.ckpt \
  --data_root_dir ./r2r \
  --dataset_name r2r \
  --lifelong_task_id task05_r2r \
  --lifelong_task_category r2r_OLN \
  --lifelong_instruction_path lifelong_data/instructions/task05_r2r.txt \
  --lifelong_memory_path lifelong_memory/skill_memory.pt \
  --lifelong_top_k 3 \
  --lifelong_lora_rank 8 \
  --lifelong_lora_alpha 16 \
  --lifelong_expansion_rank 4 \
  --lifelong_decoder_nonzero 32 \
  --lifelong_spa_beta 1e-4 \
  --batch_size 8 \
  --grad_accumulation_steps 10 \
  --max_steps 30000 \
  --save_steps 30000 \
  --run_root_dir logs/lifelong/task05_r2r


# Task6
torchrun --standalone --nnodes 1 --nproc-per-node 2 \
  vla-scripts/finetune_r2r.py \
  --use_lifelong True \
  --use_lora False \
  --vla_path ./qwbu/univla-7b \
  --lam_path ./latent_action_model/lam-stage-2.ckpt \
  --data_root_dir ./r2r \
  --dataset_name r2r \
  --lifelong_task_id task06_r2r \
  --lifelong_task_category r2r_VLN \
  --lifelong_instruction_path lifelong_data/instructions/task06_r2r.txt \
  --lifelong_memory_path lifelong_memory/skill_memory.pt \
  --lifelong_top_k 3 \
  --lifelong_lora_rank 8 \
  --lifelong_lora_alpha 16 \
  --lifelong_expansion_rank 4 \
  --lifelong_decoder_nonzero 32 \
  --lifelong_spa_beta 1e-4 \
  --batch_size 8 \
  --grad_accumulation_steps 10 \
  --max_steps 30000 \
  --save_steps 30000 \
  --run_root_dir logs/lifelong/task06_r2r


# Task7
torchrun --standalone --nnodes 1 --nproc-per-node 2 \
  vla-scripts/finetune_realworld.py \
  --use_lifelong True \
  --use_lora False \
  --vla_path ./qwbu/univla-7b \
  --lam_path ./latent_action_model/lam-stage-2.ckpt \
  --data_root_dir ./realworld \
  --dataset_name realworld \
  --lifelong_task_id task07_realworld \
  --lifelong_task_category realworld_manipulation \
  --lifelong_instruction_path lifelong_data/instructions/task07_realworld.txt \
  --lifelong_memory_path lifelong_memory/skill_memory.pt \
  --lifelong_top_k 3 \
  --lifelong_lora_rank 8 \
  --lifelong_lora_alpha 16 \
  --lifelong_expansion_rank 4 \
  --lifelong_decoder_nonzero 32 \
  --lifelong_spa_beta 1e-4 \
  --batch_size 8 \
  --grad_accumulation_steps 10 \
  --max_steps 30000 \
  --save_steps 30000 \
  --run_root_dir logs/lifelong/task07_realworld


# Task8
torchrun --standalone --nnodes 1 --nproc-per-node 2 \
  vla-scripts/finetune_libero.py \
  --use_lifelong True \
  --use_lora False \
  --vla_path ./qwbu/univla-7b \
  --lam_path ./latent_action_model/lam-stage-2.ckpt \
  --data_root_dir ./LIBERO/modified_libero_rlds \
  --dataset_name libero_spatial_no_noops \
  --lifelong_task_id task04_libero_spatial \
  --lifelong_task_category libero_manipulation \
  --lifelong_instruction_path lifelong_data/instructions/task08_libero_spatial.txt \
  --lifelong_memory_path lifelong_memory/skill_memory.pt \
  --lifelong_top_k 3 \
  --lifelong_lora_rank 8 \
  --lifelong_lora_alpha 16 \
  --lifelong_expansion_rank 4 \
  --lifelong_decoder_nonzero 32 \
  --lifelong_spa_beta 1e-4 \
  --batch_size 8 \
  --grad_accumulation_steps 10 \
  --max_steps 30000 \
  --save_steps 30000 \
  --run_root_dir logs/lifelong/task08_libero_spatial


# Task9
torchrun --standalone --nnodes 1 --nproc-per-node 2 \
  vla-scripts/finetune_r2r.py \
  --use_lifelong True \
  --use_lora False \
  --vla_path ./qwbu/univla-7b \
  --lam_path ./latent_action_model/lam-stage-2.ckpt \
  --data_root_dir ./r2r \
  --dataset_name r2r \
  --lifelong_task_id task09_r2r \
  --lifelong_task_category r2r_OLN \
  --lifelong_instruction_path lifelong_data/instructions/task09_r2r.txt \
  --lifelong_memory_path lifelong_memory/skill_memory.pt \
  --lifelong_top_k 3 \
  --lifelong_lora_rank 8 \
  --lifelong_lora_alpha 16 \
  --lifelong_expansion_rank 4 \
  --lifelong_decoder_nonzero 32 \
  --lifelong_spa_beta 1e-4 \
  --batch_size 8 \
  --grad_accumulation_steps 10 \
  --max_steps 30000 \
  --save_steps 30000 \
  --run_root_dir logs/lifelong/task09_r2r