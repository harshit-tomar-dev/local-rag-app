"""
Tests for Multi-Provider AI features, Student Study Tools, and .md/.csv Ingestion.
"""

import io
import json
import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client(tmp_path, monkeypatch):
    test_db_path = tmp_path / "study_test.db"
    test_uploads_dir = tmp_path / "uploads"
    test_uploads_dir.mkdir(parents=True, exist_ok=True)

    from app.database import Database
    from app.services import KnowledgeBaseManager
    from app.llm import LLMClient
    from app.study import StudyService

    test_db = Database(test_db_path)
    test_llm = LLMClient(provider="extractive")
    test_mgr = KnowledgeBaseManager(db=test_db, llm_client=test_llm)
    test_mgr.uploads_dir = test_uploads_dir
    test_study = StudyService(llm=test_llm)

    monkeypatch.setattr("app.main.db", test_db)
    monkeypatch.setattr("app.main.kb_manager", test_mgr)
    monkeypatch.setattr("app.main.study_service", test_study)

    with TestClient(app) as test_client:
        yield test_client


def test_list_providers(client):
    res = client.get("/api/settings/providers")
    assert res.status_code == 200
    providers = res.json()
    assert len(providers) >= 5
    ids = [p["id"] for p in providers]
    assert "openai" in ids
    assert "claude" in ids
    assert "gemini" in ids
    assert "groq" in ids
    assert "extractive" in ids


def test_verify_key_offline_provider(client):
    res = client.post("/api/settings/verify-key", json={"provider": "extractive", "api_key": ""})
    assert res.status_code == 200
    data = res.json()
    assert data["valid"] is True


def test_ingest_markdown_and_csv(client):
    kb = client.post("/kbs", json={"name": "Multi-Format Study KB"}).json()
    kb_id = kb["id"]

    # Ingest .md
    md_content = b"# Biology Study Notes\n\nPhotosynthesis converts light into chemical energy stored in glucose."
    res_md = client.post(
        f"/kbs/{kb_id}/documents",
        files=[("files", ("bio_notes.md", io.BytesIO(md_content), "text/markdown"))],
    )
    assert res_md.status_code == 200
    assert res_md.json()[0]["status"] == "indexed"

    # Ingest .csv
    csv_content = b"Topic,Formula,Units\nOhm's Law,V=IR,Volts\nKinetic Energy,KE=0.5*m*v^2,Joules"
    res_csv = client.post(
        f"/kbs/{kb_id}/documents",
        files=[("files", ("physics.csv", io.BytesIO(csv_content), "text/csv"))],
    )
    assert res_csv.status_code == 200
    assert res_csv.json()[0]["status"] == "indexed"

    # Query the markdown content
    ask_res = client.post(f"/kbs/{kb_id}/ask", json={"query": "What does photosynthesis convert?"}).json()
    assert ask_res["status"] == "answered"
    assert "chemical energy" in ask_res["answer"]


def test_study_tools_generation(client):
    kb = client.post("/kbs", json={"name": "Student Prep KB"}).json()
    kb_id = kb["id"]

    content = (
        b"Mitochondria are membrane-bound cell organelles that generate most of the chemical energy needed "
        b"to power the cell's biochemical reactions. Chemical energy produced by the mitochondria is stored in ATP. "
        b"Adenosine triphosphate (ATP) serves as the primary energy currency of cells."
    )
    client.post(
        f"/kbs/{kb_id}/documents",
        files=[("files", ("cellular_biology.txt", io.BytesIO(content), "text/plain"))],
    )

    # 1. Summary
    res_sum = client.post(
        "/api/study/generate",
        json={"kb_id": kb_id, "tool_type": "summary", "provider": "extractive"},
    )
    assert res_sum.status_code == 200
    sum_data = res_sum.json()
    assert sum_data["tool_type"] == "summary"
    assert len(sum_data["markdown"]) > 20
    assert len(sum_data["citations"]) >= 1

    # 2. Flashcards
    res_fc = client.post(
        "/api/study/generate",
        json={"kb_id": kb_id, "tool_type": "flashcards", "provider": "extractive"},
    )
    assert res_fc.status_code == 200
    fc_data = res_fc.json()
    assert fc_data["tool_type"] == "flashcards"
    assert fc_data["structured_data"] is not None
    assert len(fc_data["structured_data"]) >= 1
    assert "question" in fc_data["structured_data"][0]
    assert "answer" in fc_data["structured_data"][0]

    # 3. Practice Quiz
    res_quiz = client.post(
        "/api/study/generate",
        json={"kb_id": kb_id, "tool_type": "quiz", "provider": "extractive"},
    )
    assert res_quiz.status_code == 200
    quiz_data = res_quiz.json()
    assert quiz_data["tool_type"] == "quiz"
    assert quiz_data["structured_data"] is not None
    assert len(quiz_data["structured_data"]) >= 1
    assert "options" in quiz_data["structured_data"][0]
    assert "correct_index" in quiz_data["structured_data"][0]


def test_conversation_stream_endpoint(client):
    kb = client.post("/kbs", json={"name": "Streaming KB"}).json()
    kb_id = kb["id"]

    client.post(
        f"/kbs/{kb_id}/documents",
        files=[("files", ("guide.txt", io.BytesIO(b"Antigravity is built by DeepMind."), "text/plain"))],
    )

    conv = client.post(f"/kbs/{kb_id}/conversations", json={"title": "Stream Chat"}).json()
    conv_id = conv["id"]

    # Call streaming endpoint
    res = client.post(
        f"/conversations/{conv_id}/stream",
        json={"content": "Who built Antigravity?"},
    )
    assert res.status_code == 200
    assert "text/event-stream" in res.headers["content-type"]
    lines = res.text.split("\n")
    tokens = []
    done_seen = False
    for line in lines:
        if line.startswith("data: "):
            payload = json.loads(line[6:])
            if "token" in payload:
                tokens.append(payload["token"])
            if payload.get("done"):
                done_seen = True
    assert done_seen is True
    assert len(tokens) >= 1
