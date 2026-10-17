"""Offline-first local memory for Python AI applications."""
from .core import MemoryQA
from .packs import KnowledgePack, list_packs
from .pipeline import TextPipeline, RefinementLoop
from .associative import AssociativeMemory
from .planner import ProjectPlanner, ProjectSpec
from .brain import LocalBrain
from .federated import DomainStore, FrameEngine, AnswerFramer, FederatedBrain
from .knowledge_gate import KnowledgeGate
from .resource_router import ResourceRouter, internet_available

__version__ = "0.8.2"
__all__ = ["MemoryQA", "KnowledgePack", "list_packs", "TextPipeline", "RefinementLoop", "AssociativeMemory", "ProjectPlanner", "ProjectSpec", "LocalBrain", "DomainStore", "FrameEngine", "AnswerFramer", "FederatedBrain", "KnowledgeGate", "ResourceRouter", "internet_available"]
