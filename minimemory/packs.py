"""Portable, shareable Q&A knowledge packs."""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

PACK_FORMAT = "minimemory-pack"
PACK_VERSION = "1.2"
PACK_FILENAME = "minimemory.json"

try:
    from huggingface_hub import HfApi, hf_hub_download
except ImportError:
    HfApi = None
    hf_hub_download = None


def _check_hub_installed() -> None:
    if HfApi is None or hf_hub_download is None:
        raise ImportError("Install Hub support with: pip install 'minimemory[hub]'")


def _resolve_token(explicit: Optional[str]) -> Optional[str]:
    return explicit or os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")


class KnowledgePack:
    def __init__(self, pairs: Iterable[Tuple[str, str]], model_name: str = "all-MiniLM-L6-v2",
                 description: str = "", author: str = "", version: str = PACK_VERSION,
                 created_at: Optional[str] = None):
        self.pairs = []
        seen = set()
        for question, answer in pairs:
            if not isinstance(question, str) or not question.strip() or not isinstance(answer, str) or not answer.strip():
                raise ValueError("pack pairs require non-empty question and answer strings")
            q, a = question.strip(), answer.strip()
            if q in seen:
                self.pairs = [(old_q, old_a) for old_q, old_a in self.pairs if old_q != q]
            seen.add(q)
            self.pairs.append((q, a))
        self.model_name = str(model_name or "all-MiniLM-L6-v2")
        self.description = str(description or "")
        self.author = str(author or "")
        self.version = str(version or PACK_VERSION)
        self.created_at = created_at or datetime.now(timezone.utc).isoformat()

    @classmethod
    def from_memory(cls, ai: Any, **kwargs: Any) -> "KnowledgePack":
        model_name = kwargs.pop("model_name", getattr(ai, "model_name", "all-MiniLM-L6-v2"))
        return cls(ai.get_all_pairs(), model_name=model_name, **kwargs)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "KnowledgePack":
        if not isinstance(data, dict) or data.get("format") != PACK_FORMAT:
            raise ValueError(f"Invalid pack format: expected '{PACK_FORMAT}'")
        raw_pairs = data.get("pairs", [])
        if not isinstance(raw_pairs, list):
            raise ValueError("pack 'pairs' must be a list")
        pairs = []
        for item in raw_pairs:
            if not isinstance(item, dict) or "question" not in item or "answer" not in item:
                raise ValueError("each pack pair must contain question and answer")
            pairs.append((item["question"], item["answer"]))
        declared = data.get("pair_count")
        if declared is not None and declared != len(pairs):
            raise ValueError("pack pair_count does not match pairs")
        pack = cls(pairs, model_name=data.get("model_name", "all-MiniLM-L6-v2"),
                   description=data.get("description", ""), author=data.get("author", ""),
                   version=data.get("version", PACK_VERSION), created_at=data.get("created_at"))
        expected = data.get("checksum")
        if expected and expected != pack.checksum():
            raise ValueError("knowledge pack checksum mismatch; file may be corrupted")
        return pack

    @classmethod
    def from_file(cls, path: str) -> "KnowledgePack":
        file_path = Path(path).expanduser()
        if file_path.is_dir():
            file_path /= PACK_FILENAME
        if not file_path.is_file():
            raise FileNotFoundError(f"Knowledge pack file not found at: {file_path}")
        with file_path.open(encoding="utf-8") as handle:
            return cls.from_dict(json.load(handle))

    @classmethod
    def from_hub(cls, repo_id: str, revision: Optional[str] = None, token: Optional[str] = None) -> "KnowledgePack":
        _check_hub_installed()
        downloaded = hf_hub_download(repo_id=repo_id, filename=PACK_FILENAME, repo_type="dataset",
                                      revision=revision, token=_resolve_token(token))
        return cls.from_file(downloaded)

    def apply_to(self, ai: Any) -> int:
        return ai.learn_batch(self.pairs, source="pack", metadata={"pack_version": self.version})

    def merge_with(self, other: "KnowledgePack", new_description: Optional[str] = None) -> "KnowledgePack":
        if not isinstance(other, KnowledgePack):
            raise TypeError("other must be a KnowledgePack")
        merged = dict(self.pairs)
        merged.update(other.pairs)
        return KnowledgePack(list(merged.items()), model_name=other.model_name or self.model_name,
                             description=new_description or f"Merged: {self.description} + {other.description}",
                             author=self.author or other.author)

    def _payload_without_checksum(self) -> Dict[str, Any]:
        return {"format": PACK_FORMAT, "version": self.version, "model_name": self.model_name,
                "description": self.description, "author": self.author, "created_at": self.created_at,
                "pair_count": len(self.pairs),
                "pairs": [{"question": q, "answer": a} for q, a in self.pairs]}

    def checksum(self) -> str:
        payload = json.dumps(self._payload_without_checksum(), sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        data = self._payload_without_checksum()
        data["checksum"] = self.checksum()
        return data

    def save(self, folder: str) -> str:
        destination = Path(folder).expanduser()
        destination.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n"
        target = destination / PACK_FILENAME
        fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=str(destination), text=True)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            Path(tmp_name).replace(target)
        except Exception:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise
        (destination / "README.md").write_text(self._render_readme(), encoding="utf-8")
        return str(destination)

    def push_to_hub(self, repo_id: str, private: bool = False, token: Optional[str] = None) -> str:
        _check_hub_installed()
        api = HfApi(token=_resolve_token(token))
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=private, exist_ok=True)
        with tempfile.TemporaryDirectory() as temp_dir:
            self.save(temp_dir)
            api.upload_folder(folder_path=temp_dir, repo_id=repo_id, repo_type="dataset",
                              commit_message=f"Update KnowledgePack v{self.version} ({len(self.pairs)} pairs)")
        return f"https://huggingface.co/datasets/{repo_id}"

    def _render_readme(self) -> str:
        preview = "\n".join(f"- **{q}** → {a}" for q, a in self.pairs[:15]) or "_No pairs yet._"
        more = f"\n\n_...and {len(self.pairs) - 15} more._" if len(self.pairs) > 15 else ""
        author = f"**Author:** {self.author}\n\n" if self.author else ""
        return f"""---\ntags:\n- minimemory\n- question-answering\n- knowledge-pack\n---\n\n# {self.description or 'Minimemory Knowledge Pack'}\n\n{author}{self.description or 'A minimemory knowledge pack.'}\n\n**Model used:** `{self.model_name}`  \n**Total pairs:** {len(self.pairs)}  \n**Format version:** {self.version}  \n**SHA-256:** `{self.checksum()}`\n\n## Quick load\n\n```python\nfrom minimemory import MemoryQA\nai = MemoryQA()\nai.load_pack(\"./pack\")\n```\n\n## Preview\n\n{preview}{more}\n"""


def list_packs(search: str = "minimemory", limit: int = 20, token: Optional[str] = None) -> List[Dict[str, Any]]:
    _check_hub_installed()
    if limit < 1 or limit > 1000:
        raise ValueError("limit must be between 1 and 1000")
    api = HfApi(token=_resolve_token(token))
    results = []
    for item in api.list_datasets(search=search, limit=limit):
        results.append({"id": item.id, "downloads": getattr(item, "downloads", 0) or 0,
                        "likes": getattr(item, "likes", 0) or 0, "description": getattr(item, "description", "") or ""})
    return results
