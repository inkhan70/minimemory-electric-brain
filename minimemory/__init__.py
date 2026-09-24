"""Offline-first local memory for Python AI applications."""
from .core import MemoryQA
from .packs import KnowledgePack, list_packs
from .pipeline import TextPipeline, RefinementLoop
from .associative import AssociativeMemory
from .planner import ProjectPlanner, ProjectSpec
from .brain import LocalBrain
from .privacy import BehaviorCategory, PrivacyClass, BehaviorObservation, classify_behavior
from .router import QueryKind, QueryRoute, classify_query
from .budget import ResourceBudget
from .research import WebResearcher, Source, Evidence
from .documentation import read_documentation, update_documentation

__version__ = "0.8.0"
__all__ = ["MemoryQA", "KnowledgePack", "list_packs", "TextPipeline", "RefinementLoop", "AssociativeMemory", "ProjectPlanner", "ProjectSpec", "LocalBrain", "BehaviorCategory", "PrivacyClass", "BehaviorObservation", "classify_behavior", "QueryKind", "QueryRoute", "classify_query", "ResourceBudget", "WebResearcher", "Source", "Evidence", "read_documentation", "update_documentation"]
