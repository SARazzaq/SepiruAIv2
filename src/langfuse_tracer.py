"""
LangFuse observability wrapper — traces every LLM call with latency,
token usage, and input/output. 100% open-source (MIT license).

Usage modes:
  1. Self-hosted  → set LANGFUSE_HOST=http://localhost:3000
  2. Cloud free tier → set LANGFUSE_PUBLIC_KEY + LANGFUSE_SECRET_KEY
  3. Disabled (no config) → tracer becomes a no-op pass-through

Self-host in 1 command:
  docker run -d -p 3000:3000 langfuse/langfuse:latest
"""

import os
import time
import uuid
from typing import Optional, Generator


def _is_enabled() -> bool:
    return bool(
        os.getenv("LANGFUSE_SECRET_KEY") or
        os.getenv("LANGFUSE_PUBLIC_KEY") or
        os.getenv("LANGFUSE_HOST")
    )


def _get_client():
    """Return a LangFuse client or None if not configured."""
    if not _is_enabled():
        return None
    try:
        from langfuse import Langfuse
        return Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
    except ImportError:
        return None
    except Exception:
        return None


# ── Public API ────────────────────────────────────────────────────────────────

class LangFuseTracer:
    """
    Lightweight wrapper around LangFuse that traces LLM generations.
    Falls back to a no-op when LangFuse is not installed/configured.
    """

    def __init__(self, session_name: str = "csv-ai-analyst"):
        self._lf = _get_client()
        self._session_id = str(uuid.uuid4())
        self._session_name = session_name
        self.enabled = self._lf is not None

    def trace_generation(
        self,
        name: str,
        prompt: str,
        output: str,
        model: str = "",
        provider: str = "",
        latency_ms: int = 0,
        metadata: Optional[dict] = None,
    ) -> None:
        """Record a single LLM generation in LangFuse."""
        if not self.enabled:
            return
        try:
            trace = self._lf.trace(
                name=name,
                session_id=self._session_id,
                metadata=metadata or {},
            )
            trace.generation(
                name=name,
                model=model or provider,
                input=prompt,
                output=output,
                metadata={
                    "provider": provider,
                    "latency_ms": latency_ms,
                    **(metadata or {}),
                },
            )
            self._lf.flush()
        except Exception:
            pass  # Never break the main app

    def wrap_generate(self, ai_client, prompt: str, system: str = "", name: str = "generate") -> str:
        """Call ai_client.generate() and trace it."""
        t0 = time.time()
        result = ai_client.generate(prompt, system)
        latency = int((time.time() - t0) * 1000)
        self.trace_generation(
            name=name,
            prompt=prompt[:500],
            output=result[:500],
            model=getattr(ai_client, "model", ""),
            provider=getattr(ai_client, "provider", ""),
            latency_ms=latency,
        )
        return result

    def wrap_stream(
        self, ai_client, prompt: str, system: str = "", name: str = "stream"
    ) -> Generator[str, None, None]:
        """Call ai_client.generate_stream() and trace the full output."""
        t0 = time.time()
        full_output = []
        for chunk in ai_client.generate_stream(prompt, system):
            full_output.append(chunk)
            yield chunk
        latency = int((time.time() - t0) * 1000)
        self.trace_generation(
            name=name,
            prompt=prompt[:500],
            output="".join(full_output)[:500],
            model=getattr(ai_client, "model", ""),
            provider=getattr(ai_client, "provider", ""),
            latency_ms=latency,
        )

    def get_session_stats(self) -> dict:
        """Return session metadata (best-effort)."""
        return {
            "session_id": self._session_id,
            "enabled": self.enabled,
            "host": os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        }
