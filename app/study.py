"""
Dedicated Student Study Assistant service.
Generates comprehensive summaries, interactive flashcards, practice quizzes,
exam prep questions, formula cheat-sheets, and ELI5 (Explain Like I'm 5) breakdowns.
"""

from __future__ import annotations
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from app.llm import LLMClient
from ragcore import Chunk

logger = logging.getLogger(__name__)


STUDY_PROMPTS = {
    "summary": """You are an expert academic tutor. Create a comprehensive, crystal-clear study summary of the provided document excerpts.
Format your output in clean Markdown with the following structured sections:
# 📑 Executive Study Summary
A clear, 2-3 paragraph conceptual synthesis of what this material covers.

## 🔑 Core Concepts & Key Takeaways
- Use detailed bullet points highlighting the most crucial principles, theorems, and facts.
- Highlight key terms in **bold**.

## 📖 Key Definitions & Glossary
A quick reference table or bulleted list defining the fundamental terminology found in the text.

## 💡 Important Implications & Practical Applications
How these concepts apply in exams, problem solving, or real-world practice.""",

    "flashcards": """You are an expert exam prep coach creating active-recall flashcards for students.
Create 6 to 10 high-yield flashcards based strictly on the provided document excerpts.
CRITICAL: Respond with a valid JSON array inside a ```json ``` code block.
Each flashcard object MUST have:
- "id": integer (1, 2, 3...)
- "question": string (concise question or concept to test active recall)
- "answer": string (clear, accurate, punchy answer explaining the concept)
- "citation": string (document name and page number if available)

Example format:
```json
[
  {
    "id": 1,
    "question": "What is the primary function of mitochondria?",
    "answer": "To generate most of the chemical energy needed to power the cell's biochemical reactions via ATP synthesis.",
    "citation": "Biology_Ch4.pdf (Page 12)"
  }
]
```""",

    "quiz": """You are an expert professor designing a practice mock test for students.
Create 5 high-quality Multiple Choice Questions (MCQs) based strictly on the provided document excerpts.
CRITICAL: Respond with a valid JSON array inside a ```json ``` code block.
Each question object MUST have:
- "id": integer (1 to 5)
- "question": string (the test question)
- "options": array of 4 strings (e.g. ["A) ...", "B) ...", "C) ...", "D) ..."])
- "correct_index": integer (0 for A, 1 for B, 2 for C, 3 for D)
- "explanation": string (detailed explanation of why the correct option is right and others are wrong)
- "citation": string (document name and page number)

Example format:
```json
[
  {
    "id": 1,
    "question": "Which retrieval method does the engine use for lexical keyword matching?",
    "options": ["A) BM25", "B) TF-IDF only", "C) Word2Vec", "D) Pure Vector Search"],
    "correct_index": 0,
    "explanation": "BM25 (specifically Okapi BM25) is utilized for lexical matching with document length normalization.",
    "citation": "Architecture.pdf (Page 2)"
  }
]
```""",

    "exam_prep": """You are a senior examiner and academic coach.
Based on the provided document material, predict the top 5 most important exam questions likely to be asked on a test or final exam.
For each question provide:
1. **Question**: State the question clearly (specify marks/difficulty level: High-Yield, Conceptual, or Application).
2. **Model Answer**: A structured, complete model answer that would earn full marks.
3. **Key Points Checklist**: What the grader looks for (rubric points to hit).
4. **Common Pitfalls**: Mistakes students commonly make on this topic.
Format with clean Markdown.""",

    "eli5": """You are Richard Feynman explaining this topic to a beginner or high-school student (Explain Like I'm 5).
Rules:
1. Eliminate unnecessary academic jargon.
2. Use vivid real-world analogies, stories, and simple everyday metaphors.
3. Break the main mechanism down into simple step-by-step intuitive logic.
4. Conclude with a memorable 1-sentence "Bottom Line" summary.
Format with clean Markdown.""",

    "fact_sheet": """You are creating a 1-page high-yield Quick Revision Cheat Sheet for students before entering the exam room.
Format as clean Markdown with:
1. **Essential Formulas & Equations** (if applicable, with variable definitions).
2. **Key Theorems, Rules & Laws**.
3. **Crucial Dates, Figures, or Constants**.
4. **Do Not Forget Checklist**: Top 5 critical rules/steps to remember."""
}


class StudyService:
    def __init__(self, llm: LLMClient):
        self.llm = llm

    def generate_study_material(
        self,
        tool_type: str,
        chunks: List[Tuple[Chunk, float]],
        custom_topic: Optional[str] = None,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generate study materials (summary, flashcards, quiz, exam_prep, eli5, fact_sheet)
        from a list of document chunks.
        """
        tool_key = tool_type.lower()
        if tool_key not in STUDY_PROMPTS:
            tool_key = "summary"

        sys_prompt = STUDY_PROMPTS[tool_key]

        user_query = f"Generate {tool_key.replace('_', ' ')} for the uploaded document material."
        if custom_topic:
            user_query += f" Focus specifically on: {custom_topic}"

        # If extractive or no external LLM configured, generate high-yield fallback
        prov = (provider or self.llm.provider).lower()
        if prov == "extractive":
            return self._generate_extractive_study(tool_key, chunks, custom_topic)

        # Call active LLM
        answer_text, used_prov, used_model = self.llm.generate_answer(
            query=user_query,
            retrieved_chunks=chunks,
            provider=provider,
            model=model,
            api_key=api_key,
            system_prompt=sys_prompt,
        )

        structured_data = None
        if tool_key in ("flashcards", "quiz"):
            structured_data = self._parse_json_block(answer_text)

        # Build citations
        citations = []
        seen = set()
        for chunk, score in chunks:
            if chunk.id in seen:
                continue
            seen.add(chunk.id)
            citations.append({
                "chunk_id": chunk.id,
                "doc_id": chunk.doc_id,
                "doc_name": chunk.doc_name,
                "page_number": chunk.page_number,
                "excerpt": chunk.text[:220].strip() + ("..." if len(chunk.text) > 220 else ""),
            })

        return {
            "tool_type": tool_key,
            "provider": used_prov,
            "model": used_model,
            "markdown": answer_text,
            "structured_data": structured_data,
            "citations": citations,
        }

    def _parse_json_block(self, text: str) -> Optional[Any]:
        """Extract and parse JSON block from LLM response."""
        try:
            # First attempt: direct markdown block
            match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
            if match:
                raw_json = match.group(1).strip()
                return json.loads(raw_json)

            # Second attempt: bracket search
            bracket_match = re.search(r"(\[[\s\S]*\]|\{[\s\S]*\})", text)
            if bracket_match:
                return json.loads(bracket_match.group(1).strip())
        except Exception as e:
            logger.warning(f"Failed to parse structured JSON from study response: {e}")
        return None

    def _generate_extractive_study(
        self,
        tool_type: str,
        chunks: List[Tuple[Chunk, float]],
        custom_topic: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Offline fallback study generator without external AI."""
        if not chunks:
            return {
                "tool_type": tool_type,
                "provider": "extractive",
                "model": "extractive",
                "markdown": "No document content available to generate study material. Please upload a document first.",
                "structured_data": None,
                "citations": [],
            }

        top_chunks = [c for c, _ in chunks[:6]]
        doc_names = list(dict.fromkeys(c.doc_name for c in top_chunks))
        citations = [
            {
                "chunk_id": c.id,
                "doc_id": c.doc_id,
                "doc_name": c.doc_name,
                "page_number": c.page_number,
                "excerpt": c.text[:220].strip() + ("..." if len(c.text) > 220 else ""),
            }
            for c in top_chunks
        ]

        if tool_type == "flashcards":
            cards = []
            for idx, c in enumerate(top_chunks, 1):
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", c.text) if len(s.strip()) > 20]
                q = f"Key concept from {c.doc_name} (Page {c.page_number}):"
                a = sentences[0] if sentences else c.text[:180]
                cards.append({
                    "id": idx,
                    "question": q,
                    "answer": a,
                    "citation": f"{c.doc_name} (Page {c.page_number})",
                })
            md = "# 💡 Study Flashcards (Extractive Mode)\n\n"
            for card in cards:
                md += f"### Card {card['id']}\n**Q:** {card['question']}\n\n**A:** {card['answer']}\n\n*Source: {card['citation']}*\n\n---\n"
            return {
                "tool_type": "flashcards",
                "provider": "extractive",
                "model": "extractive",
                "markdown": md,
                "structured_data": cards,
                "citations": citations,
            }

        elif tool_type == "quiz":
            quiz = []
            for idx, c in enumerate(top_chunks[:5], 1):
                sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", c.text) if len(s.strip()) > 15]
                fact = sentences[0] if sentences else c.text[:120]
                quiz.append({
                    "id": idx,
                    "question": f"According to {c.doc_name} (Page {c.page_number}), which statement is correct?",
                    "options": [
                        f"A) {fact}",
                        "B) This statement is contradicted by the document.",
                        "C) None of the above principles apply here.",
                        "D) This concept is unrelated to the course material.",
                    ],
                    "correct_index": 0,
                    "explanation": f"Grounded directly in {c.doc_name} page {c.page_number}: '{fact}'",
                    "citation": f"{c.doc_name} (Page {c.page_number})",
                })
            md = "# 🎯 Practice Quiz (Extractive Mode)\n\n"
            for q in quiz:
                md += f"### Question {q['id']}\n{q['question']}\n\n"
                for opt in q["options"]:
                    md += f"- {opt}\n"
                md += f"\n*Correct Answer: Option A*\n*{q['explanation']}*\n\n---\n"
            return {
                "tool_type": "quiz",
                "provider": "extractive",
                "model": "extractive",
                "markdown": md,
                "structured_data": quiz,
                "citations": citations,
            }

        else:
            # Summary / ELI5 / Exam prep
            md_lines = [
                f"# 📑 Study Summary: {', '.join(doc_names)}",
                "",
                "> **Note:** Generated using Local Grounded Extractive Mode. To enable full generative AI explanations, select **OpenAI (ChatGPT)**, **Claude**, **Gemini**, or **Ollama** in the Model selector and add your API key in Settings.",
                "",
                "## 🔑 Key Excerpts by Page",
            ]
            for c in top_chunks:
                md_lines.append(f"### 📄 {c.doc_name} — Page {c.page_number}")
                md_lines.append(f"> {c.text.strip()}\n")

            return {
                "tool_type": tool_type,
                "provider": "extractive",
                "model": "extractive",
                "markdown": "\n".join(md_lines),
                "structured_data": None,
                "citations": citations,
            }
