"""Command-line interface for minimemory."""

from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .core import MemoryQA
from .packs import KnowledgePack, list_packs
from .federated import FederatedBrain


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
    out = _get_ai(args).save_pack(args.out, description=args.description, author=args.author)
    print(f"Exported memory pack to '{out}'.")


def cmd_push(args: argparse.Namespace) -> None:
    url = _get_ai(args).push_pack(args.repo_id, private=args.private, token=args.token,
                                  description=args.description, author=args.author)
    print(f"Pushed successfully: {url}")


def cmd_pull(args: argparse.Namespace) -> None:
    count = _get_ai(args).load_pack(args.source, token=args.token)
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


def cmd_federated_remember(args: argparse.Namespace) -> None:
    with FederatedBrain(args.root) as brain:
        memory_id = brain.remember(args.content, domain=args.domain, memory_type=args.memory_type,
                                   source=args.source, confidence=args.confidence, importance=args.importance)
        print(json.dumps({"domain": args.domain, "memory_id": memory_id}, ensure_ascii=False, indent=2))


def cmd_federated_qna(args: argparse.Namespace) -> None:
    with FederatedBrain(args.root) as brain:
        memory_id = brain.remember_qna(args.question, args.answer, domain=args.domain,
                                       source=args.source, confidence=args.confidence)
        print(json.dumps({"domain": args.domain, "memory_id": memory_id}, ensure_ascii=False, indent=2))


def cmd_federated_ask(args: argparse.Namespace) -> None:
    with FederatedBrain(args.root) as brain:
        domains = args.domains.split(",") if args.domains else None
        result = brain.ask(args.question, domains=domains, max_domains=args.max_domains,
                           top_k=args.top_k, min_score=args.min_score,
                           max_answer_sentences=args.max_sentences)
        print(json.dumps(result, ensure_ascii=False, indent=2))


def cmd_federated_health(args: argparse.Namespace) -> None:
    with FederatedBrain(args.root) as brain:
        result = brain.health()
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["ok"]:
            raise RuntimeError("federated memory integrity check failed")


def cmd_federated_import(args: argparse.Namespace) -> None:
    with FederatedBrain(args.root) as brain:
        result = brain.import_legacy_db(args.source_db, domain=args.domain)
        print(json.dumps(result, ensure_ascii=False, indent=2))


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

    browse = sub.add_parser("browse", help="Browse public Hub packs")
    browse.add_argument("--search", default="minimemory")
    browse.add_argument("--limit", type=int, default=20)
    browse.set_defaults(func=cmd_browse)

    f_remember = sub.add_parser("federated-remember", help="Store immutable knowledge in a domain database")
    f_remember.add_argument("content")
    f_remember.add_argument("--root", default="minimemory_data")
    f_remember.add_argument("--domain", default="memory")
    f_remember.add_argument("--memory-type", default="semantic")
    f_remember.add_argument("--source", default="user")
    f_remember.add_argument("--confidence", type=float)
    f_remember.add_argument("--importance", type=float, default=0.5)
    f_remember.set_defaults(func=cmd_federated_remember)

    f_qna = sub.add_parser("federated-qna", help="Store a Q&A pair in a domain database")
    f_qna.add_argument("question")
    f_qna.add_argument("answer")
    f_qna.add_argument("--root", default="minimemory_data")
    f_qna.add_argument("--domain", default="qna")
    f_qna.add_argument("--source", default="user")
    f_qna.add_argument("--confidence", type=float)
    f_qna.set_defaults(func=cmd_federated_qna)

    f_ask = sub.add_parser("federated-ask", help="Route a question across domain databases and frame an answer")
    f_ask.add_argument("question")
    f_ask.add_argument("--root", default="minimemory_data")
    f_ask.add_argument("--domains", help="Comma-separated domain names")
    f_ask.add_argument("--max-domains", type=int, default=3)
    f_ask.add_argument("--top-k", type=int, default=8)
    f_ask.add_argument("--min-score", type=float, default=0.05)
    f_ask.add_argument("--max-sentences", type=int, default=5)
    f_ask.set_defaults(func=cmd_federated_ask)

    f_health = sub.add_parser("federated-health", help="Check all federated databases")
    f_health.add_argument("--root", default="minimemory_data")
    f_health.set_defaults(func=cmd_federated_health)

    f_import = sub.add_parser("federated-import", help="Import an old minimemory SQLite database without modifying it")
    f_import.add_argument("source_db")
    f_import.add_argument("--root", default="minimemory_data")
    f_import.add_argument("--domain", default="memory")
    f_import.set_defaults(func=cmd_federated_import)

    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
