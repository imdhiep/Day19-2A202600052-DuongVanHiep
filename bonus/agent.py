from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
from fastembed import TextEmbedding
from feast import FeatureStore
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointStruct,
    VectorParams,
)

from app.runtime import configure_runtime

ROOT = configure_runtime()
FEAST_REPO = ROOT / "app" / "feast_repo"
FEAST_DATA = FEAST_REPO / "data"
MEMORY_COLLECTION = "bonus_memory"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
EMBED_DIM = 384


@dataclass
class MemoryChunk:
    text: str
    topic: str


class HybridMemoryAgent:
    def __init__(self) -> None:
        self.embedder = TextEmbedding(model_name=EMBED_MODEL)
        self.client = QdrantClient(":memory:")
        self.client.create_collection(
            collection_name=MEMORY_COLLECTION,
            vectors_config=VectorParams(size=EMBED_DIM, distance=Distance.COSINE),
        )
        self._seq = 0
        self._ensure_feature_store()
        self.store = FeatureStore(repo_path=str(FEAST_REPO))

    def remember(self, text: str, user_id: str = "u_001") -> None:
        chunks = self._chunk_text(text)
        vectors = list(self.embedder.embed([chunk.text for chunk in chunks]))
        points: list[PointStruct] = []
        for chunk, vector in zip(chunks, vectors):
            self._seq += 1
            points.append(
                PointStruct(
                    id=self._seq,
                    vector=vector.tolist(),
                    payload={
                        "user_id": user_id,
                        "topic": chunk.topic,
                        "text": chunk.text,
                    },
                )
            )
        self.client.upsert(collection_name=MEMORY_COLLECTION, points=points)

    def recall(self, query: str, user_id: str = "u_001") -> str:
        profile = self.store.get_online_features(
            features=[
                "user_profile_features:reading_speed_wpm",
                "user_profile_features:preferred_language",
                "user_profile_features:topic_affinity",
                "query_velocity_features:queries_last_hour",
                "query_velocity_features:distinct_topics_24h",
            ],
            entity_rows=[{"user_id": user_id}],
        ).to_dict()
        q_vec = next(self.embedder.embed([query])).tolist()
        result = self.client.query_points(
            collection_name=MEMORY_COLLECTION,
            query=q_vec,
            query_filter=Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            ),
            limit=8,
        )

        affinity = str(profile["topic_affinity"][0]).lower()
        affinity_terms = set(affinity.split())
        query_terms = set(self._tokenize(query))
        ranked = []
        for point in result.points:
            text = str(point.payload["text"])
            text_terms = set(self._tokenize(text))
            lexical = len(query_terms & text_terms) * 0.02
            affinity_boost = 0.05 if affinity_terms & text_terms else 0.0
            ranked.append((float(point.score) + lexical + affinity_boost, text))
        ranked.sort(key=lambda item: item[0], reverse=True)
        top_memories = [text for _, text in ranked[:3]]

        lines = [
            f"Question: {query}",
            (
                "User profile: prefers "
                f"{profile['preferred_language'][0]}, reads around "
                f"{profile['reading_speed_wpm'][0]} wpm, strongest affinity = "
                f"{profile['topic_affinity'][0]}."
            ),
            (
                "Recent activity: "
                f"{profile['queries_last_hour'][0]} queries in the last hour across "
                f"{profile['distinct_topics_24h'][0]} topics."
            ),
            "Top memories:",
        ]
        if top_memories:
            lines.extend(f"{idx}. {memory}" for idx, memory in enumerate(top_memories, start=1))
        else:
            lines.append("1. No episodic memories stored yet.")
        return "\n".join(lines)

    def _chunk_text(self, text: str) -> list[MemoryChunk]:
        paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
        if not paragraphs:
            paragraphs = [text.strip()]
        chunks: list[MemoryChunk] = []
        for paragraph in paragraphs:
            sentences = re.split(r"(?<=[.!?])\s+", paragraph)
            current = ""
            for sentence in sentences:
                candidate = sentence.strip()
                if not candidate:
                    continue
                if current and len(current) + len(candidate) > 280:
                    chunks.append(MemoryChunk(text=current.strip(), topic=self._infer_topic(current)))
                    current = candidate
                else:
                    current = f"{current} {candidate}".strip()
            if current:
                chunks.append(MemoryChunk(text=current, topic=self._infer_topic(current)))
        return chunks

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return re.findall(r"[a-zA-Z0-9_]+", text.lower())

    def _infer_topic(self, text: str) -> str:
        lowered = text.lower()
        rules = {
            "cloud": ["kubernetes", "autoscaling", "serverless", "lambda", "vpc", "terraform"],
            "security": ["tls", "zero-trust", "security", "iam", "secret", "oauth"],
            "ai_ml": ["embedding", "rag", "llm", "transformer"],
        }
        for topic, keywords in rules.items():
            if any(keyword in lowered for keyword in keywords):
                return topic
        return "general"

    def _ensure_feature_store(self) -> None:
        FEAST_DATA.mkdir(parents=True, exist_ok=True)
        for state_file in (FEAST_REPO / "registry.db", FEAST_REPO / "online_store.db"):
            if state_file.exists():
                state_file.unlink()
        now = datetime.now(timezone.utc).replace(microsecond=0)
        pl.DataFrame(
            {
                "user_id": ["u_001", "u_002", "u_003"],
                "reading_speed_wpm": [240, 185, 210],
                "preferred_language": ["vi-en mix", "vi", "en"],
                "topic_affinity": ["cloud security", "data engineering", "ai_ml"],
                "event_timestamp": [now - timedelta(minutes=5)] * 3,
            }
        ).write_parquet(FEAST_DATA / "user_profile.parquet")
        pl.DataFrame(
            {
                "doc_id": ["bonus_001", "bonus_002", "bonus_003"],
                "click_count_24h": [42, 27, 18],
                "ctr_7d": [0.62, 0.44, 0.33],
                "avg_dwell_seconds": [96.0, 88.0, 75.0],
                "event_timestamp": [now - timedelta(minutes=5)] * 3,
            }
        ).write_parquet(FEAST_DATA / "item_popularity.parquet")
        pl.DataFrame(
            {
                "user_id": ["u_001", "u_002", "u_003"],
                "queries_last_hour": [6, 2, 4],
                "distinct_topics_24h": [3, 2, 4],
                "event_timestamp": [now - timedelta(minutes=5)] * 3,
            }
        ).write_parquet(FEAST_DATA / "query_velocity.parquet")
        subprocess.run(["feast", "apply"], cwd=str(FEAST_REPO), check=True, capture_output=True, text=True)
        subprocess.run(
            ["feast", "materialize-incremental", now.strftime("%Y-%m-%dT%H:%M:%S")],
            cwd=str(FEAST_REPO),
            check=True,
            capture_output=True,
            text=True,
        )
