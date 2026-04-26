
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


def load_instruction_samples(path_or_text: Optional[Union[str, Path]]) -> List[str]:

    if path_or_text is None:
        return []

    path = Path(path_or_text)
    if path.exists():
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text())
            if isinstance(payload, dict):
                payload = payload.get("instructions", [])
            return [str(item) for item in payload]

        instructions = []
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            if path.suffix.lower() == ".jsonl":
                record = json.loads(line)
                instructions.append(str(record.get("instruction", record.get("text", line))))
            else:
                instructions.append(line)
        return instructions

    return [part.strip() for part in str(path_or_text).split("||") if part.strip()]


@dataclass
class SkillMemoryEntry:
    task_id: str
    category: str
    expert_group: List[str]
    head_name: str
    semantic_basis: torch.Tensor

    @property
    def projection(self) -> torch.Tensor:
        return self.semantic_basis @ self.semantic_basis.transpose(0, 1)


class CLIPTextEmbedder:

    def __init__(self, model_name: str = "openai/clip-vit-base-patch32", device: Optional[torch.device] = None) -> None:
        self.model_name = model_name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._processor = None

    def _load(self) -> None:
        if self._model is not None:
            return
        from transformers import CLIPModel, CLIPProcessor

        self._processor = CLIPProcessor.from_pretrained(self.model_name)
        self._model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        self._model.eval()

    @torch.inference_mode()
    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        if not texts:
            raise ValueError("TSDA needs at least one instruction to build or query a semantic subspace.")
        self._load()
        inputs = self._processor(text=list(texts), padding=True, truncation=True, return_tensors="pt").to(self.device)
        features = self._model.get_text_features(**inputs)
        return F.normalize(features.float(), dim=-1).cpu()


class SkillMemoryBank:

    def __init__(self, entries: Optional[Iterable[SkillMemoryEntry]] = None) -> None:
        self.entries: Dict[str, SkillMemoryEntry] = {}
        for entry in entries or []:
            self.entries[entry.task_id] = entry

    def __len__(self) -> int:
        return len(self.entries)

    def categories(self) -> List[str]:
        return sorted({entry.category for entry in self.entries.values()})

    def is_new_category(self, category: str) -> bool:
        return category not in {entry.category for entry in self.entries.values()}

    def upsert(
        self,
        task_id: str,
        category: str,
        instructions: Sequence[str],
        expert_group: Sequence[str],
        head_name: str,
        embedder: CLIPTextEmbedder,
        semantic_rank: int,
    ) -> SkillMemoryEntry:
        embeddings = embedder.encode(instructions)
        basis = self._build_semantic_basis(embeddings, semantic_rank)
        entry = SkillMemoryEntry(
            task_id=str(task_id),
            category=str(category),
            expert_group=[str(expert_id) for expert_id in expert_group],
            head_name=str(head_name),
            semantic_basis=basis,
        )
        self.entries[entry.task_id] = entry
        return entry

    def retrieve(
        self,
        instruction: str,
        embedder: CLIPTextEmbedder,
        top_k: int,
    ) -> List[Tuple[str, float, List[str], str]]:
        if not self.entries:
            return []
        query = embedder.encode([instruction])[0]
        scored = []
        for entry in self.entries.values():
            projected = entry.projection @ query
            score = F.cosine_similarity(query, projected, dim=0, eps=1e-8).item()
            scored.append((entry.task_id, score, entry.expert_group, entry.head_name))
        scored.sort(key=lambda item: item[1], reverse=True)
        return scored[:top_k]

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = []
        for entry in self.entries.values():
            payload.append(
                {
                    "task_id": entry.task_id,
                    "category": entry.category,
                    "expert_group": entry.expert_group,
                    "head_name": entry.head_name,
                    "semantic_basis": entry.semantic_basis.cpu(),
                }
            )
        torch.save({"entries": payload}, path)

    @classmethod
    def load(cls, path: Optional[Union[str, Path]]) -> "SkillMemoryBank":
        if path is None:
            return cls()
        path = Path(path)
        if not path.exists():
            return cls()
        payload = torch.load(path, map_location="cpu")
        entries = []
        for item in payload.get("entries", []):
            entries.append(
                SkillMemoryEntry(
                    task_id=str(item["task_id"]),
                    category=str(item["category"]),
                    expert_group=[str(expert_id) for expert_id in item["expert_group"]],
                    head_name=str(item["head_name"]),
                    semantic_basis=item["semantic_basis"].float(),
                )
            )
        return cls(entries)

    @staticmethod
    def _build_semantic_basis(embeddings: torch.Tensor, semantic_rank: int) -> torch.Tensor:
        centered = embeddings.float()
        centered = centered - centered.mean(dim=0, keepdim=True)
        if centered.shape[0] == 1:
            centered = embeddings.float()
        _, _, vh = torch.linalg.svd(centered, full_matrices=False)
        rank = max(1, min(semantic_rank, vh.shape[0]))
        basis = vh[:rank].transpose(0, 1).contiguous()
        return F.normalize(basis, dim=0)
