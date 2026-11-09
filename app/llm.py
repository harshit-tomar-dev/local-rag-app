"""
LLM Provider integrations: OpenAI, Claude (Anthropic), Gemini, Groq, OpenRouter, Ollama, and Extractive.
Includes streaming SSE generators, timeout handling, retries, and clean HTTP error mappings.
"""

from __future__ import annotations
import json
import logging
import os
import re
import time
from typing import Any, AsyncGenerator, Dict, Generator, List, Optional, Tuple
import httpx

from app.config import settings
from ragcore import Chunk, _tokenize, STOP_WORDS

logger = logging.getLogger(__name__)


class LLMError(Exception):
    """Raised when an external LLM provider fails."""
    def __init__(self, message: str, status_code: int = 502):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


PROVIDER_CONFIGS = {
    "openai": {
        "display_name": "OpenAI (ChatGPT)",
        "base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
        "models": ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo", "o1-mini", "o3-mini"],
        "needs_key": True,
    },
    "claude": {
        "display_name": "Anthropic Claude",
        "base_url": "https://api.anthropic.com/v1",
        "default_model": "claude-3-5-sonnet-20241022",
        "env_key": "ANTHROPIC_API_KEY",
        "models": ["claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
        "needs_key": True,
    },
    "anthropic": {
        "display_name": "Anthropic Claude",
        "base_url": "https://api.anthropic.com/v1",
        "default_model": "claude-3-5-sonnet-20241022",
        "env_key": "ANTHROPIC_API_KEY",
        "models": ["claude-3-7-sonnet-20250219", "claude-3-5-sonnet-20241022", "claude-3-5-haiku-20241022", "claude-3-opus-20240229"],
        "needs_key": True,
    },
    "gemini": {
        "display_name": "Google Gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "default_model": "gemini-2.0-flash",
        "env_key": "GEMINI_API_KEY",
        "models": ["gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        "needs_key": True,
    },
    "groq": {
        "display_name": "Groq (Ultra-Fast)",
        "base_url": "https://api.groq.com/openai/v1",
        "default_model": "llama-3.3-70b-versatile",
        "env_key": "GROQ_API_KEY",
        "models": ["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768"],
        "needs_key": True,
    },
    "openrouter": {
        "display_name": "OpenRouter",
        "base_url": "https://openrouter.ai/api/v1",
        "default_model": "deepseek/deepseek-chat",
        "env_key": "OPENROUTER_API_KEY",
        "models": ["deepseek/deepseek-chat", "openai/gpt-4o", "anthropic/claude-3.5-sonnet"],
        "needs_key": True,
    },
    "deepseek": {
        "display_name": "DeepSeek",
        "base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",
        "env_key": "DEEPSEEK_API_KEY",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "needs_key": True,
    },
    "ollama": {
        "display_name": "Ollama (Local AI)",
        "base_url": "http://localhost:11434",
        "default_model": "llama3.2",
        "env_key": None,
        "models": ["llama3.2", "mistral", "deepseek-r1", "phi3", "nomic-embed-text"],
        "needs_key": False,
    },
    "openai_compatible": {
        "display_name": "Custom OpenAI-Compatible",
        "base_url": "http://localhost:11434/v1",
        "default_model": "llama3.2",
        "env_key": "LLM_API_KEY",
        "models": ["default"],
        "needs_key": False,
    },
    "extractive": {
        "display_name": "Extractive (Offline / No Key)",
        "base_url": "",
        "default_model": "extractive",
        "env_key": None,
        "models": ["extractive"],
        "needs_key": False,
    },
}


class LLMClient:
    def __init__(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.provider = (provider or settings.LLM_PROVIDER).lower()
        cfg = PROVIDER_CONFIGS.get(self.provider, PROVIDER_CONFIGS["extractive"])
        self.model = model or settings.LLM_MODEL or cfg["default_model"]

        default_base_url = cfg["base_url"] or settings.LLM_BASE_URL
        self.base_url = (base_url or default_base_url).rstrip("/")

        # Resolve API Key: explicit param -> config env key -> global LLM_API_KEY
        resolved_key = api_key if api_key is not None else ""
        if not resolved_key and cfg.get("env_key"):
            resolved_key = getattr(settings, cfg["env_key"], "") or os.getenv(cfg["env_key"], "")
        if not resolved_key:
            resolved_key = settings.LLM_API_KEY
        self.api_key = resolved_key.strip()

    def _get_active_credentials(
        self,
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> Tuple[str, str, str, str]:
        prov = (provider or self.provider).lower()
        if prov == "anthropic":
            prov = "claude"
        cfg = PROVIDER_CONFIGS.get(prov, PROVIDER_CONFIGS["extractive"])

        m = model or self.model or cfg["default_model"]
        b = (base_url or self.base_url or cfg["base_url"]).rstrip("/")

        k = (api_key if api_key is not None else self.api_key).strip()
        if not k and cfg.get("env_key"):
            k = (getattr(settings, cfg["env_key"], "") or os.getenv(cfg["env_key"], "")).strip()
        if not k:
            k = settings.LLM_API_KEY.strip()

        if cfg["needs_key"] and not k:
            raise LLMError(
                f"API Key required for {cfg['display_name']}. Please enter your API key in Settings (or .env).",
                status_code=400,
            )

        return prov, m, b, k

    def generate_answer(
        self,
        query: str,
        retrieved_chunks: List[Tuple[Chunk, float]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Tuple[str, str, str]:
        """
        Generate answer based on retrieved context chunks.
        Returns (answer_text, provider_name, model_name).
        Raises LLMError on external provider outage, timeout, or authentication issues.
        """
        prov, m, b, k = self._get_active_credentials(provider, model, api_key)

        # Extractive fallback
        if prov == "extractive":
            if not retrieved_chunks or all(s <= 0.0 for _, s in retrieved_chunks):
                return (
                    "No relevant context found in the selected documents for this question.",
                    prov,
                    "extractive",
                )
            return self._generate_extractive(query, retrieved_chunks)

        # Context construction
        relevant = [(c, s) for c, s in retrieved_chunks if s > 0.0] if retrieved_chunks else []
        is_general_knowledge = not relevant

        if prov in ("openai", "gemini", "groq", "openrouter", "deepseek", "openai_compatible"):
            return self._call_openai_chat(query, relevant, prov, m, b, k, system_prompt, is_general_knowledge)
        elif prov in ("claude", "anthropic"):
            return self._call_claude_messages(query, relevant, prov, m, b, k, system_prompt, is_general_knowledge)
        elif prov == "ollama":
            return self._call_ollama(query, relevant, prov, m, b, system_prompt, is_general_knowledge)
        else:
            return self._generate_extractive(query, retrieved_chunks)

    def _generate_extractive(
        self,
        query: str,
        retrieved_chunks: List[Tuple[Chunk, float]],
    ) -> Tuple[str, str, str]:
        """Grounded extractive answer without hallucination."""
        best_chunk, top_score = retrieved_chunks[0]

        q_tokens = set(_tokenize(query)) - STOP_WORDS
        if not q_tokens:
            q_tokens = set(_tokenize(query))

        raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?\n])\s+", best_chunk.text) if s.strip()]
        scored_sentences = []
        for sent in raw_sentences:
            s_tokens = set(_tokenize(sent))
            overlap = len(q_tokens.intersection(s_tokens))
            if overlap > 0:
                scored_sentences.append((sent, overlap))

        if scored_sentences:
            scored_sentences.sort(key=lambda x: x[1], reverse=True)
            best_sentence = scored_sentences[0][0]
            answer = (
                f"**[Extractive Mode]** Grounded answer from **{best_chunk.doc_name}** (Page {best_chunk.page_number}):\n\n"
                f"> \"{best_sentence}\"\n\n"
                f"*(Full excerpt available in citations below)*"
            )
        else:
            answer = (
                f"**[Extractive Mode]** Grounded answer from **{best_chunk.doc_name}** (Page {best_chunk.page_number}):\n\n"
                f"> \"{best_chunk.text.strip()}\""
            )
        return answer, "extractive", "extractive"

    # ------------------ OpenAI & Compatible Providers ------------------

    def _call_openai_chat(
        self,
        query: str,
        relevant_chunks: List[Tuple[Chunk, float]],
        provider: str,
        model: str,
        base_url: str,
        api_key: str,
        system_prompt: Optional[str] = None,
        general_knowledge: bool = False,
    ) -> Tuple[str, str, str]:
        if general_knowledge:
            sys = system_prompt or "You are an intelligent, helpful AI assistant like ChatGPT or Claude. Answer clearly, accurately, and politely."
            user_content = query
        else:
            context_str = "\n\n---\n\n".join(
                f"Source: {c.doc_name} (Page {c.page_number})\nExcerpt: {c.text}"
                for c, _ in relevant_chunks
            )
            sys = system_prompt or (
                "You are an expert AI study assistant and tutor like ChatGPT. "
                "Answer the user's question accurately, thoroughly, and clearly based on the provided document excerpts. "
                "Always cite the source document name and page number where relevant. "
                "Format using clean markdown with bolding, lists, and code blocks if appropriate."
            )
            user_content = f"Document Context Excerpts:\n{context_str}\n\nQuestion: {query}"

        messages = [
            {"role": "system", "content": sys},
            {"role": "user", "content": user_content},
        ]

        url = f"{base_url}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        payload = {
            "model": model,
            "messages": messages,
            "temperature": 0.2 if not general_knowledge else 0.7,
        }

        last_error = None
        for attempt in range(2):
            try:
                with httpx.Client(timeout=45.0) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        text = data["choices"][0]["message"]["content"].strip()
                        return text, provider, model
                    elif resp.status_code == 401:
                        raise LLMError(f"Authentication failed for {provider}: Invalid API Key. Please update your key.", status_code=401)
                    elif resp.status_code == 429:
                        raise LLMError(f"Rate limit or quota exceeded for {provider}. Check your API usage and billing.", status_code=429)
                    else:
                        raise LLMError(
                            f"{provider.capitalize()} returned HTTP {resp.status_code}: {resp.text}",
                            status_code=502,
                        )
            except httpx.ConnectError:
                last_error = LLMError(f"Unable to connect to {provider} at {base_url}. Check network or endpoint.", status_code=503)
            except httpx.TimeoutException:
                last_error = LLMError(f"Request to {provider} timed out after 45s for model '{model}'.", status_code=504)
            except LLMError as e:
                raise e
            except Exception as e:
                last_error = LLMError(f"Unexpected error communicating with {provider}: {e}", status_code=502)

            time.sleep(0.5)

        if last_error:
            raise last_error
        raise LLMError(f"Failed to reach {provider}.", status_code=503)

    # ------------------ Claude (Anthropic) ------------------

    def _call_claude_messages(
        self,
        query: str,
        relevant_chunks: List[Tuple[Chunk, float]],
        provider: str,
        model: str,
        base_url: str,
        api_key: str,
        system_prompt: Optional[str] = None,
        general_knowledge: bool = False,
    ) -> Tuple[str, str, str]:
        if general_knowledge:
            sys = system_prompt or "You are Claude, a helpful, intelligent, polite, and comprehensive AI assistant."
            user_content = query
        else:
            context_str = "\n\n---\n\n".join(
                f"Source: {c.doc_name} (Page {c.page_number})\nExcerpt: {c.text}"
                for c, _ in relevant_chunks
            )
            sys = system_prompt or (
                "You are Claude, an expert AI tutor and study assistant. "
                "Use the provided document excerpts to give an insightful, comprehensive, and well-structured answer. "
                "Include exact citations with document names and page numbers."
            )
            user_content = f"Context:\n{context_str}\n\nQuestion: {query}"

        url = f"{base_url}/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }

        payload: Dict[str, Any] = {
            "model": model,
            "max_tokens": 4096,
            "messages": [{"role": "user", "content": user_content}],
            "temperature": 0.2 if not general_knowledge else 0.7,
        }
        if sys:
            payload["system"] = sys

        last_error = None
        for attempt in range(2):
            try:
                with httpx.Client(timeout=45.0) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        parts = [block["text"] for block in data.get("content", []) if block.get("type") == "text"]
                        return "".join(parts).strip(), provider, model
                    elif resp.status_code == 401:
                        raise LLMError("Anthropic Authentication failed: Invalid API Key. Please check your key in Settings.", status_code=401)
                    elif resp.status_code == 429:
                        raise LLMError("Anthropic rate limit or quota exceeded. Check your plan at console.anthropic.com.", status_code=429)
                    else:
                        raise LLMError(f"Claude returned HTTP {resp.status_code}: {resp.text}", status_code=502)
            except httpx.ConnectError:
                last_error = LLMError("Unable to connect to Anthropic Claude servers.", status_code=503)
            except httpx.TimeoutException:
                last_error = LLMError(f"Claude request timed out after 45s for model '{model}'.", status_code=504)
            except LLMError as e:
                raise e
            except Exception as e:
                last_error = LLMError(f"Error communicating with Claude: {e}", status_code=502)

            time.sleep(0.5)

        if last_error:
            raise last_error
        raise LLMError("Failed to reach Anthropic Claude.", status_code=503)

    # ------------------ Ollama ------------------

    def _call_ollama(
        self,
        query: str,
        relevant_chunks: List[Tuple[Chunk, float]],
        provider: str,
        model: str,
        base_url: str,
        system_prompt: Optional[str] = None,
        general_knowledge: bool = False,
    ) -> Tuple[str, str, str]:
        if general_knowledge:
            prompt = (
                f"{system_prompt or 'You are an intelligent, helpful AI assistant.'}\n\n"
                f"User Question: {query}\n\nAnswer:"
            )
        else:
            context_str = "\n\n---\n\n".join(
                f"Source: {c.doc_name} (Page {c.page_number})\nExcerpt: {c.text}"
                for c, _ in relevant_chunks
            )
            prompt = (
                f"{system_prompt or 'You are an intelligent, helpful AI study assistant equipped with document knowledge.'}\n\n"
                f"Document Context:\n{context_str}\n\n"
                f"User Question: {query}\n\n"
                "Instructions:\n"
                "- Ground your answer in the document context.\n"
                "- Clearly mention document names and page numbers.\n\n"
                "Answer:"
            )

        url = f"{base_url}/api/generate"
        payload = {"model": model, "prompt": prompt, "stream": False}

        last_error = None
        for attempt in range(2):
            try:
                with httpx.Client(timeout=60.0) as client:
                    resp = client.post(url, json=payload)
                    if resp.status_code == 200:
                        data = resp.json()
                        return data.get("response", "").strip(), provider, model
                    else:
                        raise LLMError(f"Ollama returned HTTP {resp.status_code}: {resp.text}", status_code=502)
            except httpx.ConnectError:
                last_error = LLMError(f"Unable to connect to Ollama at {base_url}. Ensure Ollama is running ('ollama serve').", status_code=503)
            except httpx.TimeoutException:
                last_error = LLMError(f"Ollama timed out after 60s for model '{model}'.", status_code=504)
            except LLMError as e:
                last_error = e
            except Exception as e:
                last_error = LLMError(f"Unexpected error communicating with Ollama: {e}", status_code=502)

            time.sleep(0.5)

        if last_error:
            raise last_error
        raise LLMError("Failed to reach Ollama provider.", status_code=503)

    # ------------------ Streaming Token Generator ------------------

    def stream_answer(
        self,
        query: str,
        retrieved_chunks: List[Tuple[Chunk, float]],
        provider: Optional[str] = None,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> Generator[str, None, None]:
        """
        Yields tokens/chunks as they are generated for real-time streaming.
        """
        prov, m, b, k = self._get_active_credentials(provider, model, api_key)

        # Extractive streaming
        if prov == "extractive" or not prov:
            full_ans, _, _ = self.generate_answer(query, retrieved_chunks, provider=prov, model=m, api_key=k)
            # Break into word pieces to emulate typing
            words = full_ans.split(" ")
            for i, w in enumerate(words):
                yield w + (" " if i < len(words) - 1 else "")
                time.sleep(0.015)
            return

        # Prepare messages
        relevant = [(c, s) for c, s in retrieved_chunks if s > 0.0] if retrieved_chunks else []
        if relevant:
            context_str = "\n\n---\n\n".join(
                f"Source: {c.doc_name} (Page {c.page_number})\nExcerpt: {c.text}"
                for c, _ in relevant
            )
            sys = system_prompt or "You are an expert AI tutor like ChatGPT. Ground your answer in the provided documents and cite sources."
            user_content = f"Context:\n{context_str}\n\nQuestion: {query}"
        else:
            sys = system_prompt or "You are an intelligent, helpful AI assistant like ChatGPT."
            user_content = query

        # OpenAI-compatible streaming
        if prov in ("openai", "gemini", "groq", "openrouter", "deepseek", "openai_compatible"):
            url = f"{b}/chat/completions"
            headers = {"Content-Type": "application/json"}
            if k:
                headers["Authorization"] = f"Bearer {k}"

            payload = {
                "model": m,
                "messages": [
                    {"role": "system", "content": sys},
                    {"role": "user", "content": user_content},
                ],
                "temperature": 0.2,
                "stream": True,
            }

            try:
                with httpx.Client(timeout=60.0) as client:
                    with client.stream("POST", url, headers=headers, json=payload) as resp:
                        if resp.status_code != 200:
                            err_body = resp.read().decode("utf-8", errors="replace")
                            raise LLMError(f"{prov.capitalize()} stream error ({resp.status_code}): {err_body}", status_code=resp.status_code)

                        for line in resp.iter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            raw = line[6:].strip()
                            if raw == "[DONE]":
                                break
                            try:
                                chunk_json = json.loads(raw)
                                delta = chunk_json.get("choices", [{}])[0].get("delta", {})
                                content = delta.get("content")
                                if content:
                                    yield content
                            except Exception:
                                continue
            except Exception as e:
                logger.error(f"Streaming error with {prov}: {e}")
                # Fallback non-stream
                ans, _, _ = self.generate_answer(query, retrieved_chunks, provider=prov, model=m, api_key=k, system_prompt=sys)
                yield ans

        elif prov in ("claude", "anthropic"):
            # Claude streaming
            url = f"{b}/messages"
            headers = {
                "x-api-key": k,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload = {
                "model": m,
                "max_tokens": 4096,
                "system": sys,
                "messages": [{"role": "user", "content": user_content}],
                "stream": True,
            }
            try:
                with httpx.Client(timeout=60.0) as client:
                    with client.stream("POST", url, headers=headers, json=payload) as resp:
                        if resp.status_code != 200:
                            err_body = resp.read().decode("utf-8", errors="replace")
                            raise LLMError(f"Claude stream error: {err_body}", status_code=resp.status_code)
                        for line in resp.iter_lines():
                            if not line or not line.startswith("data: "):
                                continue
                            raw = line[6:].strip()
                            try:
                                chunk_json = json.loads(raw)
                                if chunk_json.get("type") == "content_block_delta":
                                    text = chunk_json.get("delta", {}).get("text", "")
                                    if text:
                                        yield text
                            except Exception:
                                continue
            except Exception:
                ans, _, _ = self.generate_answer(query, retrieved_chunks, provider=prov, model=m, api_key=k, system_prompt=sys)
                yield ans

        elif prov == "ollama":
            url = f"{b}/api/generate"
            payload = {"model": m, "prompt": f"{sys}\n\n{user_content}", "stream": True}
            try:
                with httpx.Client(timeout=60.0) as client:
                    with client.stream("POST", url, json=payload) as resp:
                        for line in resp.iter_lines():
                            if not line:
                                continue
                            try:
                                cj = json.loads(line)
                                token = cj.get("response", "")
                                if token:
                                    yield token
                                if cj.get("done", False):
                                    break
                            except Exception:
                                continue
            except Exception:
                ans, _, _ = self.generate_answer(query, retrieved_chunks, provider=prov, model=m, api_key=k, system_prompt=sys)
                yield ans

    def verify_key(self, provider: str, api_key: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Verify if an API key is valid by making a lightweight test query."""
        prov = provider.lower()
        if prov in ("extractive", "ollama"):
            return {"valid": True, "provider": prov, "message": f"{prov.capitalize()} does not require an API key."}

        cfg = PROVIDER_CONFIGS.get(prov)
        if not cfg:
            raise LLMError(f"Unsupported provider '{provider}'.", status_code=400)

        test_model = model or cfg["default_model"]
        b = cfg["base_url"]

        if prov in ("openai", "gemini", "groq", "openrouter", "deepseek"):
            url = f"{b}/chat/completions"
            headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
            payload = {
                "model": test_model,
                "messages": [{"role": "user", "content": "ping"}],
                "max_tokens": 5,
            }
            try:
                with httpx.Client(timeout=15.0) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    if resp.status_code == 200:
                        return {"valid": True, "provider": prov, "model": test_model, "message": "Key is valid and active!"}
                    elif resp.status_code == 401:
                        return {"valid": False, "provider": prov, "message": "Invalid API key."}
                    else:
                        return {"valid": False, "provider": prov, "message": f"Server responded with {resp.status_code}: {resp.text[:120]}"}
            except Exception as e:
                return {"valid": False, "provider": prov, "message": str(e)}

        elif prov in ("claude", "anthropic"):
            url = f"{b}/messages"
            headers = {
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload = {
                "model": test_model,
                "max_tokens": 5,
                "messages": [{"role": "user", "content": "ping"}],
            }
            try:
                with httpx.Client(timeout=15.0) as client:
                    resp = client.post(url, headers=headers, json=payload)
                    if resp.status_code == 200:
                        return {"valid": True, "provider": "claude", "model": test_model, "message": "Key is valid and active!"}
                    elif resp.status_code == 401:
                        return {"valid": False, "provider": "claude", "message": "Invalid API key."}
                    else:
                        return {"valid": False, "provider": "claude", "message": f"Server responded with {resp.status_code}: {resp.text[:120]}"}
            except Exception as e:
                return {"valid": False, "provider": "claude", "message": str(e)}

        return {"valid": True, "provider": prov, "message": "Ready."}
