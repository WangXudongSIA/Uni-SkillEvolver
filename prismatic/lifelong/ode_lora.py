
from __future__ import annotations

import math
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


TargetModules = Union[str, Sequence[str]]


@dataclass
class ODELoraConfig:

    rank: int = 32
    alpha: int = 16
    dropout: float = 0.0
    expansion_rank: int = 8
    decoder_nonzero: int = 32
    coactivation_top_k: int = 2
    target_modules: TargetModules = "all-linear"
    exclude_modules: Tuple[str, ...] = ("lm_head",)
    adapter_dtype: torch.dtype = torch.float32
    freeze_base: bool = True
    train_bias: bool = False
    tisi_enabled: bool = True
    tisi_temperature: float = 1.0
    seed: int = 0
    metadata: Dict[str, str] = field(default_factory=dict)

    @property
    def scaling(self) -> float:
        return self.alpha / max(1, self.rank)


def real_fourier_orthogonal_multiply(coefficients: torch.Tensor) -> torch.Tensor:

    if coefficients.shape[-1] == 0:
        return coefficients

    output_dtype = coefficients.dtype
    n = coefficients.shape[-1]
    work = coefficients.float()
    spectrum = torch.zeros(
        *work.shape[:-1],
        n // 2 + 1,
        dtype=torch.complex64,
        device=work.device,
    )

    sqrt_n = math.sqrt(n)
    spectrum[..., 0] = torch.complex(work[..., 0] * sqrt_n, torch.zeros_like(work[..., 0]))

    pair_scale = math.sqrt(n / 2.0)
    max_frequency = (n - 1) // 2
    column = 1
    for frequency in range(1, max_frequency + 1):
        cos_coeff = work[..., column]
        sin_coeff = work[..., column + 1]
        spectrum[..., frequency] = torch.complex(cos_coeff * pair_scale, -sin_coeff * pair_scale)
        column += 2

    if n % 2 == 0:
        nyquist = work[..., -1] * sqrt_n
        spectrum[..., -1] = torch.complex(nyquist, torch.zeros_like(nyquist))

    transformed = torch.fft.irfft(spectrum, n=n, dim=-1)
    return transformed.to(output_dtype)


def _kaiming_parameter(shape: Tuple[int, ...], dtype: torch.dtype, device: torch.device) -> nn.Parameter:
    parameter = nn.Parameter(torch.empty(shape, dtype=dtype, device=device))
    nn.init.kaiming_uniform_(parameter, a=math.sqrt(5))
    return parameter


def _sample_sparse_mask(
    shape: Tuple[int, int],
    nonzero: int,
    used_mask: Optional[torch.Tensor],
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    total = shape[0] * shape[1]
    requested = max(0, min(nonzero, total))
    mask = torch.zeros(total, dtype=torch.bool)
    if requested == 0:
        return mask.view(shape).to(device)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    if used_mask is not None:
        available = torch.nonzero(~used_mask.detach().cpu().bool().flatten(), as_tuple=False).flatten()
        if available.numel() >= requested:
            selected = available[torch.randperm(available.numel(), generator=generator)[:requested]]
        else:
            selected = torch.randperm(total, generator=generator)[:requested]
    else:
        selected = torch.randperm(total, generator=generator)[:requested]

    mask[selected] = True
    return mask.view(shape).to(device)


def _stable_int_hash(*parts: object) -> int:
    digest = hashlib.sha256("::".join(str(part) for part in parts).encode("utf-8")).hexdigest()
    return int(digest[:16], 16)


def _module_dict_key(identifier: str) -> str:
    return str(identifier).replace(".", "__dot__").replace("/", "__slash__")


class MaskedDecoderBlock(nn.Module):

    def __init__(
        self,
        out_features: int,
        rank: int,
        mask: torch.Tensor,
        dtype: torch.dtype,
        device: torch.device,
        trainable: bool,
    ) -> None:
        super().__init__()
        self.coefficients = _kaiming_parameter((out_features, rank), dtype=dtype, device=device)
        self.coefficients.requires_grad_(trainable)
        self.register_buffer("mask", mask.to(device=device, dtype=dtype), persistent=True)
        with torch.no_grad():
            self.coefficients.mul_(self.mask)

    @classmethod
    def zero_padded(
        cls,
        out_features: int,
        rank: int,
        dtype: torch.dtype,
        device: torch.device,
    ) -> "MaskedDecoderBlock":
        block = cls(
            out_features=out_features,
            rank=rank,
            mask=torch.zeros(out_features, rank, dtype=torch.bool),
            dtype=dtype,
            device=device,
            trainable=False,
        )
        with torch.no_grad():
            block.coefficients.zero_()
        return block

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return F.linear(encoded, self.coefficients * self.mask)

    def set_trainable(self, trainable: bool) -> None:
        self.coefficients.requires_grad_(trainable)


class ODELoraDecoderExpert(nn.Module):

    def __init__(self, expert_id: str, out_features: int) -> None:
        super().__init__()
        self.expert_id = str(expert_id)
        self.out_features = out_features
        self.blocks = nn.ModuleList()

    def add_block(self, block: MaskedDecoderBlock) -> None:
        self.blocks.append(block)

    def set_trainable(self, trainable: bool) -> None:
        for block in self.blocks:
            block.set_trainable(trainable)

    def forward(self, encoded_blocks: Sequence[torch.Tensor]) -> torch.Tensor:
        if len(encoded_blocks) != len(self.blocks):
            raise ValueError(
                f"Expert {self.expert_id} received {len(encoded_blocks)} encoder blocks, "
                f"but owns {len(self.blocks)} decoder blocks."
            )

        decoded = None
        for encoded, block in zip(encoded_blocks, self.blocks):
            contribution = block(encoded)
            decoded = contribution if decoded is None else decoded + contribution

        if decoded is None:
            raise RuntimeError(f"Expert {self.expert_id} has no decoder blocks.")
        return real_fourier_orthogonal_multiply(decoded)


class ODELoraLinear(nn.Module):

    def __init__(self, base_layer: nn.Linear, config: ODELoraConfig, module_name: str) -> None:
        super().__init__()
        self.base_layer = base_layer
        self.config = config
        self.module_name = module_name
        self.in_features = base_layer.in_features
        self.out_features = base_layer.out_features
        self.dropout = nn.Dropout(config.dropout)
        self.encoder_blocks = nn.ParameterList()
        self.decoder_experts = nn.ModuleDict()
        self.tisi_gates = nn.ModuleDict()
        self.used_masks: List[torch.Tensor] = []
        self.active_expert_ids: List[str] = []
        self.current_expert_id: Optional[str] = None
        self._last_tisi_gate: Optional[torch.Tensor] = None

        if config.freeze_base:
            self.base_layer.weight.requires_grad_(False)
            if self.base_layer.bias is not None:
                self.base_layer.bias.requires_grad_(config.train_bias)

    @property
    def adapter_device(self) -> torch.device:
        return self.base_layer.weight.device

    @property
    def adapter_dtype(self) -> torch.dtype:
        return self.config.adapter_dtype

    def add_task(
        self,
        task_id: str,
        expand_encoder: bool,
        train_current: bool = True,
    ) -> None:
        task_id = _module_dict_key(str(task_id))

        for encoder in self.encoder_blocks:
            encoder.requires_grad_(False)
        for expert in self.decoder_experts.values():
            expert.set_trainable(False)
        for gate in self.tisi_gates.values():
            gate.requires_grad_(False)

        if len(self.encoder_blocks) == 0:
            self._append_encoder_block(self.config.rank, trainable=True)
        elif expand_encoder:
            self._append_encoder_block(self.config.expansion_rank, trainable=True)

        if task_id not in self.decoder_experts:
            self._append_decoder_expert(task_id, trainable=train_current)
        else:
            self.decoder_experts[task_id].set_trainable(train_current)

        if self.config.tisi_enabled:
            if task_id not in self.tisi_gates:
                gate = nn.Linear(self.in_features, 2, dtype=self.adapter_dtype, device=self.adapter_device)
                self.tisi_gates[task_id] = gate
            self.tisi_gates[task_id].requires_grad_(train_current)

    def set_active_experts(self, expert_ids: Sequence[str], current_expert_id: Optional[str]) -> None:
        existing = [
            _module_dict_key(str(expert_id))
            for expert_id in expert_ids
            if _module_dict_key(str(expert_id)) in self.decoder_experts
        ]
        self.active_expert_ids = existing
        self.current_expert_id = _module_dict_key(str(current_expert_id)) if current_expert_id is not None else None

    def tisi_loss(self) -> Optional[torch.Tensor]:
        return self._last_tisi_gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base = self.base_layer(x)
        self._last_tisi_gate = None

        if not self.active_expert_ids or len(self.encoder_blocks) == 0:
            return base

        adapter_input = self.dropout(x).to(self.adapter_dtype)
        encoded_blocks = [F.linear(adapter_input, encoder) for encoder in self.encoder_blocks]

        update = None
        for expert_id in self.active_expert_ids:
            expert_update = self.decoder_experts[expert_id](encoded_blocks)
            if self.config.tisi_enabled and expert_id == self.current_expert_id and expert_id in self.tisi_gates:
                gate = self._compute_tisi_gate(x, expert_id)
                expert_update = expert_update * gate.to(dtype=expert_update.dtype)

            update = expert_update if update is None else update + expert_update

        if update is None:
            return base
        return base + (update * self.config.scaling).to(dtype=base.dtype)

    def _append_encoder_block(self, rank: int, trainable: bool) -> None:
        encoder = _kaiming_parameter((rank, self.in_features), dtype=self.adapter_dtype, device=self.adapter_device)
        encoder.requires_grad_(trainable)
        self.encoder_blocks.append(encoder)
        self.used_masks.append(torch.zeros(self.out_features, rank, dtype=torch.bool, device=self.adapter_device))

        for expert in self.decoder_experts.values():
            expert.add_block(
                MaskedDecoderBlock.zero_padded(
                    out_features=self.out_features,
                    rank=rank,
                    dtype=self.adapter_dtype,
                    device=self.adapter_device,
                )
            )

    def _append_decoder_expert(self, task_id: str, trainable: bool) -> None:
        expert = ODELoraDecoderExpert(task_id, self.out_features)
        total_rank = sum(encoder.shape[0] for encoder in self.encoder_blocks)

        for block_idx, encoder in enumerate(self.encoder_blocks):
            rank = encoder.shape[0]
            block_nonzero = math.ceil(self.config.decoder_nonzero * rank / max(1, total_rank))
            seed = self.config.seed + (_stable_int_hash(self.module_name, task_id, block_idx) % 2_000_000_000)
            mask = _sample_sparse_mask(
                shape=(self.out_features, rank),
                nonzero=block_nonzero,
                used_mask=self.used_masks[block_idx],
                seed=seed,
                device=self.adapter_device,
            )
            self.used_masks[block_idx] = torch.logical_or(self.used_masks[block_idx], mask.bool())
            expert.add_block(
                MaskedDecoderBlock(
                    out_features=self.out_features,
                    rank=rank,
                    mask=mask,
                    dtype=self.adapter_dtype,
                    device=self.adapter_device,
                    trainable=trainable,
                )
            )

        self.decoder_experts[task_id] = expert

    def _compute_tisi_gate(self, x: torch.Tensor, expert_id: str) -> torch.Tensor:
        gate_layer = self.tisi_gates[expert_id]
        gate_input = x.detach() if not self.training else x
        gate_input = gate_input.to(self.adapter_dtype)

        if gate_input.dim() == 1:
            embedding = gate_input.unsqueeze(0)
        else:
            embedding = gate_input.reshape(gate_input.shape[0], -1, self.in_features).mean(dim=1)

        logits = gate_layer(embedding)
        if self.training:
            gate_probs = F.gumbel_softmax(logits, tau=self.config.tisi_temperature, hard=True, dim=-1)
            activation = gate_probs[:, 1]
        else:
            activation = (logits.argmax(dim=-1) == 1).to(logits.dtype)

        self._last_tisi_gate = activation.mean()
        view_shape = [activation.shape[0]] + [1] * (x.dim() - 1)
        return activation.view(*view_shape)


def _iter_parent_modules(model: nn.Module) -> Iterable[Tuple[nn.Module, str, str, nn.Module]]:
    for parent_name, parent in model.named_modules():
        for child_name, child in list(parent.named_children()):
            full_name = f"{parent_name}.{child_name}" if parent_name else child_name
            yield parent, child_name, full_name, child


def _matches_target(full_name: str, target_modules: TargetModules) -> bool:
    if target_modules == "all-linear":
        return True
    if isinstance(target_modules, str):
        targets = (target_modules,)
    else:
        targets = tuple(target_modules)
    return any(full_name == target or full_name.endswith(f".{target}") or target in full_name for target in targets)


def inject_ode_lora(model: nn.Module, config: ODELoraConfig) -> Dict[str, int]:

    replacements = 0
    adapted_parameters = 0
    for parent, child_name, full_name, child in list(_iter_parent_modules(model)):
        if isinstance(child, ODELoraLinear):
            continue
        if not isinstance(child, nn.Linear):
            continue
        if any(excluded and excluded in full_name for excluded in config.exclude_modules):
            continue
        if not _matches_target(full_name, config.target_modules):
            continue

        wrapped = ODELoraLinear(child, config=config, module_name=full_name)
        setattr(parent, child_name, wrapped)
        replacements += 1
        adapted_parameters += child.in_features * config.rank + config.decoder_nonzero

    return {"replaced_linear_layers": replacements, "approx_initial_adapter_parameters": adapted_parameters}


def configure_ode_lora_for_task(
    model: nn.Module,
    task_id: str,
    expand_encoder: bool,
    retrieved_expert_ids: Optional[Sequence[str]] = None,
) -> List[str]:

    retrieved = [str(expert_id) for expert_id in (retrieved_expert_ids or []) if str(expert_id) != str(task_id)]
    active = [str(task_id)] + retrieved

    configured = 0
    for module in model.modules():
        if isinstance(module, ODELoraLinear):
            module.add_task(task_id=task_id, expand_encoder=expand_encoder, train_current=True)
            module.set_active_experts(active, current_expert_id=task_id)
            configured += 1

    if configured == 0:
        raise RuntimeError("No ODELoraLinear modules found. Call inject_ode_lora before task configuration.")

    return active


def activate_ode_lora_experts(
    model: nn.Module,
    expert_ids: Sequence[str],
    current_expert_id: Optional[str] = None,
) -> List[str]:

    active = [str(expert_id) for expert_id in expert_ids]
    current = current_expert_id or (active[0] if active else None)
    for module in model.modules():
        if isinstance(module, ODELoraLinear):
            module.set_active_experts(active, current_expert_id=current)
    return active


def get_ode_lora_sparsity_loss(model: nn.Module) -> Optional[torch.Tensor]:
    losses = [module.tisi_loss() for module in model.modules() if isinstance(module, ODELoraLinear)]
    losses = [loss for loss in losses if loss is not None]
    if not losses:
        return None
    return torch.stack([loss.float() for loss in losses]).sum()


def count_ode_lora_parameters(model: nn.Module) -> Dict[str, int]:
    trainable = 0
    total = 0
    layers = 0
    for module in model.modules():
        if isinstance(module, ODELoraLinear):
            layers += 1
            for name, parameter in module.named_parameters():
                if name.startswith("base_layer."):
                    continue
                total += parameter.numel()
                if parameter.requires_grad:
                    trainable += parameter.numel()
    return {"ode_lora_layers": layers, "ode_lora_parameters": total, "ode_lora_trainable_parameters": trainable}


def save_ode_lora_state(model: nn.Module, path: Union[str, Path], metadata: Optional[Dict[str, object]] = None) -> None:

    adapter_state = {}
    for name, module in model.named_modules():
        if isinstance(module, ODELoraLinear):
            adapter_state[name] = {
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in module.state_dict().items()
                    if not key.startswith("base_layer.")
                },
                "used_masks": [mask.detach().cpu() for mask in module.used_masks],
                "active_expert_ids": list(module.active_expert_ids),
                "current_expert_id": module.current_expert_id,
            }

    payload = {"metadata": metadata or {}, "adapter_state": adapter_state}
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_ode_lora_state(
    model: nn.Module,
    path: Union[str, Path],
    strict: bool = False,
) -> Dict[str, object]:

    payload = torch.load(path, map_location="cpu")
    modules = dict(model.named_modules())
    missing = []
    for name, saved in payload.get("adapter_state", {}).items():
        module = modules.get(name)
        if not isinstance(module, ODELoraLinear):
            missing.append(name)
            continue

        state_dict = saved.get("state_dict", saved)
        module.load_state_dict(state_dict, strict=strict)
        if "used_masks" in saved:
            module.used_masks = [mask.to(device=module.adapter_device, dtype=torch.bool) for mask in saved["used_masks"]]
        if "active_expert_ids" in saved:
            module.active_expert_ids = list(saved["active_expert_ids"])
        if "current_expert_id" in saved:
            module.current_expert_id = saved["current_expert_id"]

    metadata = dict(payload.get("metadata", {}))
    metadata["missing_ode_lora_modules"] = missing
    return metadata
