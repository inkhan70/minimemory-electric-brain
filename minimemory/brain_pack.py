"""Portable `.mmpack` exchange format for the associative local brain.

Only knowledge is exported by default. Behavior history is intentionally not
included. Embeddings are omitted so the receiving installation can regenerate
machine-local semantic indexes with its own model.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

FORMAT = "minimemory-mmpack"
VERSION = "1.0"


def _fingerprint(subject: str, key: str, value: str) -> str:
    raw = "\x1f".join(x.strip().casefold() for x in (subject, key, value))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_json(zf: zipfile.ZipFile, name: str) -> list | dict:
    return json.loads(zf.read(name).decode("utf-8"))


def export_brain(db_path: str, destination: str, *, include_behavior: bool = False) -> str:
    if include_behavior:
        raise ValueError("behavior export is disabled by default; sensitive behavior is never included in .mmpack")
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    try:
        q = [dict(r) for r in db.execute("SELECT question,answer,source,confidence,metadata_json,created_at,updated_at FROM knowledge_memory ORDER BY id")]
        memories = [dict(r) for r in db.execute("SELECT id,memory_type,content,source,confidence,importance,metadata_json,created_at,updated_at,access_count FROM assoc_memories ORDER BY id")]
        entities = [dict(r) for r in db.execute("SELECT entity_key,name,entity_type,description,confidence,metadata_json,created_at,updated_at FROM assoc_entities ORDER BY id")]
        aliases = [dict(r) for r in db.execute("SELECT e.entity_key,a.alias FROM assoc_aliases a JOIN assoc_entities e ON e.id=a.entity_id ORDER BY e.entity_key,a.alias")]
        relations = [dict(r) for r in db.execute("SELECT s.entity_key AS subject,r.predicate,o.entity_key AS object_key,r.object_text,r.source,r.confidence,r.metadata_json,r.created_at FROM assoc_relations r JOIN assoc_entities s ON s.id=r.subject_id LEFT JOIN assoc_entities o ON o.id=r.object_id ORDER BY r.id")]
        links = [dict(r) for r in db.execute("SELECT m.id AS memory_id,e.entity_key,l.relation FROM assoc_links l JOIN assoc_memories m ON m.id=l.memory_id JOIN assoc_entities e ON e.id=l.entity_id ORDER BY m.id")]
        for row in q + memories + entities + aliases + relations + links:
            for key in list(row):
                if key.endswith("_json"):
                    try: row[key[:-5]] = json.loads(row.pop(key) or "{}")
                    except json.JSONDecodeError: row[key[:-5]] = {}
        manifest = {
            "format": FORMAT, "version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(),
            "knowledge_qna": len(q), "associative_memories": len(memories), "entities": len(entities),
            "relations": len(relations), "embeddings": "excluded; regenerate locally", "behavior": "excluded",
        }
        payloads = {
            "manifest.json": manifest, "qna.json": q, "memories.json": memories,
            "entities.json": entities, "aliases.json": aliases, "relations.json": relations, "links.json": links,
        }
        checksums = {}
        for name, payload in payloads.items():
            if name == "manifest.json":
                continue
            raw = (json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
            checksums[name] = hashlib.sha256(raw).hexdigest()
        manifest["checksums"] = checksums
        target = Path(destination).expanduser()
        target.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(prefix=target.name + ".", suffix=".tmp", dir=target.parent, delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            with zipfile.ZipFile(tmp_path, "w", zipfile.ZIP_DEFLATED) as zf:
                for name, payload in payloads.items():
                    if name == "manifest.json":
                        continue
                    zf.writestr(name, json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
                zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n")
            tmp_path.replace(target)
        finally:
            if tmp_path.exists(): tmp_path.unlink()
        return str(target)
    finally:
        db.close()


def inspect_brain_pack(path: str) -> Dict[str, Any]:
    with zipfile.ZipFile(path, "r") as zf:
        if "manifest.json" not in zf.namelist(): raise ValueError("invalid .mmpack: missing manifest.json")
        manifest = _load_json(zf, "manifest.json")
        if manifest.get("format") != FORMAT: raise ValueError("unsupported .mmpack format")
        for name, expected in manifest.get("checksums", {}).items():
            actual = hashlib.sha256(zf.read(name)).hexdigest()
            if actual != expected: raise ValueError(f"checksum mismatch: {name}")
        return manifest


def import_brain(db_path: str, path: str) -> Dict[str, int]:
    manifest = inspect_brain_pack(path)
    with zipfile.ZipFile(path, "r") as zf:
        qna = _load_json(zf, "qna.json")
        memories = _load_json(zf, "memories.json")
        entities = _load_json(zf, "entities.json")
        aliases = _load_json(zf, "aliases.json")
        relations = _load_json(zf, "relations.json")
        links = _load_json(zf, "links.json")
    db = sqlite3.connect(db_path)
    try:
        qcount = mcount = ecount = rcount = 0
        for row in qna:
            db.execute("INSERT INTO knowledge_memory(question,answer,usage_count,source,confidence,metadata_json,created_at,updated_at) VALUES(?,?,1,?,?,?,?,?) ON CONFLICT(question) DO UPDATE SET answer=excluded.answer,source=excluded.source,confidence=excluded.confidence,metadata_json=excluded.metadata_json,updated_at=CURRENT_TIMESTAMP", (row["question"], row["answer"], row.get("source","pack"), row.get("confidence"), json.dumps(row.get("metadata",{}), ensure_ascii=False, sort_keys=True), row.get("created_at"), row.get("updated_at")))
            qcount += 1
        entity_map = {}
        for row in entities:
            db.execute("INSERT INTO assoc_entities(entity_key,name,entity_type,description,confidence,metadata_json,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(entity_key) DO UPDATE SET name=excluded.name,entity_type=excluded.entity_type,description=excluded.description,confidence=excluded.confidence,metadata_json=excluded.metadata_json,updated_at=excluded.updated_at", (row["entity_key"],row["name"],row.get("entity_type","unknown"),row.get("description",""),row.get("confidence"),json.dumps(row.get("metadata",{}), ensure_ascii=False, sort_keys=True),row.get("created_at"),row.get("updated_at")))
            entity_map[row["entity_key"]] = db.execute("SELECT id FROM assoc_entities WHERE entity_key=?", (row["entity_key"],)).fetchone()[0]
            ecount += 1
        for row in aliases:
            if row["entity_key"] in entity_map: db.execute("INSERT OR IGNORE INTO assoc_aliases(entity_id,alias) VALUES(?,?)", (entity_map[row["entity_key"]],row["alias"]))
        memory_map = {}
        for row in memories:
            normalized = " ".join(row["content"].casefold().split())
            existing = db.execute("SELECT id FROM assoc_memories WHERE memory_type=? AND normalized=? AND source=?", (row["memory_type"],normalized,row.get("source","pack"))).fetchone()
            if existing:
                mid = int(existing[0])
                db.execute("UPDATE assoc_memories SET confidence=COALESCE(?,confidence),importance=?,metadata_json=?,updated_at=? WHERE id=?", (row.get("confidence"),row.get("importance",0.5),json.dumps(row.get("metadata",{}), ensure_ascii=False, sort_keys=True),row.get("updated_at"),mid))
            else:
                cur = db.execute("INSERT INTO assoc_memories(memory_type,content,normalized,source,confidence,importance,metadata_json,created_at,updated_at,access_count) VALUES(?,?,?,?,?,?,?,?,?,?)", (row["memory_type"],row["content"],normalized,row.get("source","pack"),row.get("confidence"),row.get("importance",0.5),json.dumps(row.get("metadata",{}), ensure_ascii=False, sort_keys=True),row.get("created_at"),row.get("updated_at"),row.get("access_count",0)))
                mid = int(cur.lastrowid); mcount += 1
            memory_map[int(row["id"])] = mid
        for row in relations:
            sid = entity_map.get(row["subject"]); oid = entity_map.get(row.get("object_key")) if row.get("object_key") else None
            if sid:
                db.execute("INSERT INTO assoc_relations(subject_id,predicate,object_id,object_text,source,confidence,metadata_json,created_at) VALUES(?,?,?,?,?,?,?,?)", (sid,row["predicate"],oid,row.get("object_text"),row.get("source","pack"),row.get("confidence"),json.dumps(row.get("metadata",{}), ensure_ascii=False, sort_keys=True),row.get("created_at"))); rcount += 1
        for row in links:
            mid = memory_map.get(int(row["memory_id"])); eid = entity_map.get(row["entity_key"])
            if mid and eid: db.execute("INSERT OR IGNORE INTO assoc_links(memory_id,entity_id,relation) VALUES(?,?,?)", (mid,eid,row.get("relation","mentions")))
        db.commit()
        return {"qna_imported": qcount, "memories_added": mcount, "entities_processed": ecount, "relations_added": rcount, "behavior_imported": 0, "format_version": manifest.get("version")}
    finally:
        db.close()
