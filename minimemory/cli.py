"""Command-line interface for minimemory."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .core import MemoryQA
from .packs import KnowledgePack, list_packs
from .ai import OllamaAdapter, OpenAICompatibleAdapter
from .brain import LocalBrain


def _get_ai(args: argparse.Namespace) -> MemoryQA:
    return MemoryQA(db_path=args.db)



def cmd_learn(args: argparse.Namespace) -> None:
    with _get_ai(args) as ai:
        ai.learn(args.question, args.answer, source=args.source, confidence=args.confidence)
        print("Learned 1 memory.")

def cmd_ask(args: argparse.Namespace) -> None:
    with _get_ai(args) as ai:
        print(json.dumps(ai.ask(args.question, threshold=args.threshold, top_k=args.top_k), ensure_ascii=False, indent=2))

def cmd_search(args: argparse.Namespace) -> None:
    with _get_ai(args) as ai:
        print(json.dumps(ai.search(args.question, top_k=args.top_k, min_score=args.min_score, source=args.source), ensure_ascii=False, indent=2))

def cmd_stats(args: argparse.Namespace) -> None:
    with _get_ai(args) as ai:
        print(json.dumps(ai.stats(), ensure_ascii=False, indent=2))

def cmd_health(args: argparse.Namespace) -> None:
    with _get_ai(args) as ai:
        result = ai.health()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["ok"]:
            raise RuntimeError("database integrity check failed")

def cmd_forget(args: argparse.Namespace) -> None:
    with _get_ai(args) as ai:
        print("Deleted." if ai.delete(args.memory_id) else "Memory not found.")

def cmd_export(args: argparse.Namespace) -> None:
    with _get_ai(args) as ai:
        out = ai.save_pack(args.out, description=args.description, author=args.author)
    print(f"Exported memory pack to '{out}'.")


def cmd_push(args: argparse.Namespace) -> None:
    with _get_ai(args) as ai:
        url = ai.push_pack(args.repo_id, private=args.private, token=args.token,
                           description=args.description, author=args.author)
    print(f"Pushed successfully: {url}")


def cmd_pull(args: argparse.Namespace) -> None:
    with _get_ai(args) as ai:
        count = ai.load_pack(args.source, token=args.token)
    print(f"Ingested {count} pairs from '{args.source}'.")


def cmd_merge(args: argparse.Namespace) -> None:
    if len(args.sources) < 2:
        raise ValueError("please specify at least 2 pack sources to merge")
    merged = KnowledgePack.from_file(args.sources[0])
    for source in args.sources[1:]:
        merged = merged.merge_with(KnowledgePack.from_file(source))
    if args.description:
        merged.description = args.description
    print(f"Merged {len(args.sources)} packs into '{merged.save(args.out)}' ({len(merged.pairs)} unique pairs).")



def _make_ai_adapter(args: argparse.Namespace):
    if args.provider == "ollama":
        return OllamaAdapter(model=args.model, base_url=args.base_url, timeout=args.timeout)
    if args.provider == "openai-compatible":
        return OpenAICompatibleAdapter(model=args.model, base_url=args.base_url, api_key=args.api_key, timeout=args.timeout)
    raise ValueError(f"unsupported AI provider: {args.provider}")

def cmd_ai_test(args: argparse.Namespace) -> None:
    adapter = _make_ai_adapter(args)
    result = adapter.test()
    print(json.dumps(result, ensure_ascii=False, indent=2))

def cmd_ai_ask(args: argparse.Namespace) -> None:
    adapter = _make_ai_adapter(args)
    with _get_ai(args) as ai:
        result = ai.chat(args.question, adapter, threshold=args.memory_threshold,
                          auto_learn=args.auto_learn, approve=args.approve,
                          confidence=args.confidence, source=f"ai:{adapter.provider}")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_brain_observe(args: argparse.Namespace) -> None:
    with LocalBrain(args.db, behavior_tracking=args.enable) as brain:
        if args.clear:
            print(json.dumps(brain.clear_behavior(), ensure_ascii=False, indent=2))
            return
        if args.enable is not None:
            brain.set_behavior_tracking(args.enable)
        if args.query:
            print(json.dumps(brain.observe(args.query), ensure_ascii=False, indent=2))
        else:
            print(json.dumps(brain.behavior_status(), ensure_ascii=False, indent=2))

def cmd_brain_evolve(args: argparse.Namespace) -> None:
    with LocalBrain(args.db, behavior_tracking=args.enable) as brain:
        print(json.dumps(brain.evolve(min_accesses=args.min_accesses), ensure_ascii=False, indent=2))

def cmd_brain_pulse(args: argparse.Namespace) -> None:
    with LocalBrain(args.db, behavior_tracking=args.enable) as brain:
        print(json.dumps(brain.pulse(), ensure_ascii=False, indent=2))

def cmd_brain_export(args: argparse.Namespace) -> None:
    with LocalBrain(args.db) as brain:
        print(brain.export_knowledge(args.out))

def cmd_brain_import(args: argparse.Namespace) -> None:
    with LocalBrain(args.db) as brain:
        print(json.dumps(brain.import_knowledge(args.source), ensure_ascii=False, indent=2))

def cmd_brain_pack_inspect(args: argparse.Namespace) -> None:
    with LocalBrain(args.db) as brain:
        print(json.dumps(brain.inspect_knowledge_pack(args.source), ensure_ascii=False, indent=2))

def cmd_browse(args: argparse.Namespace) -> None:
    packs = list_packs(search=args.search, limit=args.limit, token=args.token)
    if not packs:
        print("No knowledge packs found.")
        return
    print(f"{'REPO ID':<35} | {'DOWNLOADS':<10} | {'LIKES':<6} | DESCRIPTION")
    print("-" * 100)
    for pack in packs:
        print(f"{pack['id']:<35} | {pack['downloads']:<10} | {pack['likes']:<6} | {pack['description']}")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="minimemory", description="Local Q&A memory with knowledge packs.")
    parser.add_argument("--db", default="memory.db", help="SQLite database path")
    parser.add_argument("--token", help="Hugging Face authentication token")
    sub = parser.add_subparsers(dest="cmd", required=True)

    learn = sub.add_parser("learn", help="Add or update one memory")
    learn.add_argument("question")
    learn.add_argument("answer")
    learn.add_argument("--source", default="user")
    learn.add_argument("--confidence", type=float)
    learn.set_defaults(func=cmd_learn)

    ask = sub.add_parser("ask", help="Ask the local memory")
    ask.add_argument("question")
    ask.add_argument("--threshold", type=float, default=0.55)
    ask.add_argument("--top-k", type=int, default=5)
    ask.set_defaults(func=cmd_ask)

    search = sub.add_parser("search", help="Search stored memories")
    search.add_argument("question")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--min-score", type=float, default=0.0)
    search.add_argument("--source")
    search.set_defaults(func=cmd_search)

    stats = sub.add_parser("stats", help="Show memory statistics")
    stats.set_defaults(func=cmd_stats)

    health = sub.add_parser("health", help="Check SQLite integrity")
    health.set_defaults(func=cmd_health)

    forget = sub.add_parser("forget", help="Delete a memory by ID")
    forget.add_argument("memory_id", type=int)
    forget.set_defaults(func=cmd_forget)

    export = sub.add_parser("export", help="Export current memory to a local pack")
    export.add_argument("out")
    export.add_argument("--description", default="")
    export.add_argument("--author", default="")
    export.set_defaults(func=cmd_export)

    push = sub.add_parser("push", help="Push current memory to the Hub")
    push.add_argument("repo_id")
    push.add_argument("--private", action="store_true")
    push.add_argument("--description", default="")
    push.add_argument("--author", default="")
    push.set_defaults(func=cmd_push)

    pull = sub.add_parser("pull", help="Pull a local or Hub pack into memory")
    pull.add_argument("source")
    pull.set_defaults(func=cmd_pull)

    merge = sub.add_parser("merge", help="Merge local packs")
    merge.add_argument("sources", nargs="+")
    merge.add_argument("--out", required=True)
    merge.add_argument("--description", default="")
    merge.set_defaults(func=cmd_merge)

    ai = sub.add_parser("ai", help="Connect minimemory to an AI model server")
    ai_sub = ai.add_subparsers(dest="ai_cmd", required=True)

    ai_common = argparse.ArgumentParser(add_help=False)
    ai_common.add_argument("--provider", choices=["ollama", "openai-compatible"], default="ollama")
    ai_common.add_argument("--model", default="llama3.2:3b")
    ai_common.add_argument("--base-url", default="http://127.0.0.1:11434")
    ai_common.add_argument("--api-key", default=None)
    ai_common.add_argument("--timeout", type=float, default=120.0)

    ai_test = ai_sub.add_parser("test", parents=[ai_common], help="Test the AI connection")
    ai_test.set_defaults(func=cmd_ai_test)

    ai_ask = ai_sub.add_parser("ask", parents=[ai_common], help="Ask an AI with retrieved local memory")
    ai_ask.add_argument("question")
    ai_ask.add_argument("--memory-threshold", type=float, default=0.75)
    ai_ask.add_argument("--confidence", type=float)
    ai_ask.add_argument("--auto-learn", action=argparse.BooleanOptionalAction, default=True)
    ai_ask.add_argument("--approve", action="store_true", help="Approve the AI answer for learning")
    ai_ask.set_defaults(func=cmd_ai_ask)

    brain = sub.add_parser("brain", help="Electric-brain evolution, privacy, and knowledge exchange")
    brain_sub = brain.add_subparsers(dest="brain_cmd", required=True)

    observe = brain_sub.add_parser("observe", help="Classify/optionally store a behavior signal")
    observe.add_argument("query", nargs="?")
    observe.add_argument("--enable", action=argparse.BooleanOptionalAction, default=None)
    observe.add_argument("--clear", action="store_true")
    observe.set_defaults(func=cmd_brain_observe)

    evolve = brain_sub.add_parser("evolve", help="Run a bounded memory evolution/consolidation pulse")
    evolve.add_argument("--min-accesses", type=int, default=2)
    evolve.add_argument("--enable", action=argparse.BooleanOptionalAction, default=False)
    evolve.set_defaults(func=cmd_brain_evolve)

    pulse = brain_sub.add_parser("pulse", help="Show electric-brain health and evolution state")
    pulse.add_argument("--enable", action=argparse.BooleanOptionalAction, default=False)
    pulse.set_defaults(func=cmd_brain_pulse)

    brain_export = brain_sub.add_parser("export", help="Export portable .mmpack knowledge")
    brain_export.add_argument("out")
    brain_export.set_defaults(func=cmd_brain_export)

    brain_import = brain_sub.add_parser("import", help="Import and merge a .mmpack knowledge pack")
    brain_import.add_argument("source")
    brain_import.set_defaults(func=cmd_brain_import)

    brain_inspect = brain_sub.add_parser("inspect-pack", help="Verify a .mmpack without importing it")
    brain_inspect.add_argument("source")
    brain_inspect.set_defaults(func=cmd_brain_pack_inspect)

    browse = sub.add_parser("browse", help="Browse public Hub packs")
    browse.add_argument("--search", default="minimemory")
    browse.add_argument("--limit", type=int, default=20)
    browse.set_defaults(func=cmd_browse)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
