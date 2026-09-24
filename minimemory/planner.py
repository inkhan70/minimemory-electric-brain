"""Dependency-aware lightweight project planning helpers."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

@dataclass
class ProjectSpec:
    goal: str
    platform: Optional[str] = None
    frontend: Optional[str] = None
    backend: Optional[str] = None
    database: Optional[str] = None
    language: Optional[str] = None
    requirements: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)

class ProjectPlanner:
    """Create deterministic project plans before an LLM generates code."""
    def analyze(self, goal: str, *, platform: Optional[str] = None, frontend: Optional[str] = None,
                backend: Optional[str] = None, database: Optional[str] = None,
                language: Optional[str] = None) -> Dict[str, Any]:
        if not isinstance(goal, str) or not goal.strip():
            raise ValueError("goal must be a non-empty string")
        missing = []
        if not platform: missing.append("platform")
        if not (frontend or language): missing.append("frontend_or_language")
        return {
            "spec": ProjectSpec(goal.strip(), platform, frontend, backend, database, language).__dict__,
            "missing": missing,
            "tasks": self._tasks(platform, frontend, backend, database, language),
            "status": "needs_clarification" if missing else "ready"
        }

    @staticmethod
    def _tasks(platform, frontend, backend, database, language):
        tasks = ["define requirements", "choose compatible components", "create project structure"]
        if frontend or language: tasks.append("implement user interface or entry point")
        if backend: tasks.append("implement backend/API")
        if database: tasks.append("implement persistence and migrations")
        tasks += ["connect interfaces", "validate dependencies and APIs", "run tests", "review provenance and licenses"]
        return tasks
