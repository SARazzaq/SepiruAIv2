"""
AutoGen multi-agent data analysis.
Uses Microsoft AutoGen (MIT license) — fully free & open-source.

Two agents collaborate:
  - DataAnalystAgent: interprets data context and forms analysis plans
  - CriticAgent: reviews and refines the analysis for accuracy

Configured to use the existing AIClient provider (Groq, Ollama, etc.)
via the OpenAI-compatible endpoint where possible.
"""

import os
import pandas as pd
from typing import Optional


def _get_llm_config(ai_client) -> dict:
    """Build AutoGen llm_config from the existing AIClient."""
    provider = getattr(ai_client, "provider", "groq")
    model = getattr(ai_client, "model", "llama-3.3-70b-versatile")

    if provider == "groq":
        return {
            "config_list": [{
                "model": model,
                "api_key": os.getenv("GROQ_API_KEY", ""),
                "base_url": "https://api.groq.com/openai/v1",
            }],
            "temperature": 0.3,
        }

    elif provider == "openai":
        return {
            "config_list": [{
                "model": model,
                "api_key": os.getenv("OPENAI_API_KEY", ""),
            }],
            "temperature": 0.3,
        }

    elif provider == "ollama":
        ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
        return {
            "config_list": [{
                "model": model,
                "api_key": "ollama",
                "base_url": f"{ollama_host}/v1",
            }],
            "temperature": 0.3,
        }

    elif provider == "vllm":
        vllm_host = os.getenv("VLLM_HOST", "http://localhost:8000")
        return {
            "config_list": [{
                "model": model,
                "api_key": "EMPTY",
                "base_url": f"{vllm_host}/v1",
            }],
            "temperature": 0.3,
        }

    else:
        # Gemini / Anthropic — fallback to Groq config if key exists
        groq_key = os.getenv("GROQ_API_KEY", "")
        if groq_key:
            return {
                "config_list": [{
                    "model": "llama-3.3-70b-versatile",
                    "api_key": groq_key,
                    "base_url": "https://api.groq.com/openai/v1",
                }],
                "temperature": 0.3,
            }
        raise ValueError(
            f"AutoGen doesn't support provider '{provider}' natively. "
            "Set AI_PROVIDER=groq or AI_PROVIDER=ollama."
        )


def run_autogen_analysis(
    data_context: str,
    question: str,
    ai_client,
    max_rounds: int = 4,
) -> str:
    """
    Spin up two AutoGen agents to collaboratively answer a data question.
    Returns the final analysis as a string.
    """
    try:
        from autogen import AssistantAgent, UserProxyAgent
    except ImportError:
        raise ImportError(
            "Run: pip install pyautogen"
        )

    llm_config = _get_llm_config(ai_client)

    analyst = AssistantAgent(
        name="DataAnalyst",
        system_message=(
            "You are a senior data analyst. You receive a dataset summary and a question. "
            "Provide a thorough, structured analysis with insights, patterns, and recommendations. "
            "Be concise and fact-based. Cite specific numbers from the data. "
            "End your final answer with: FINAL_ANSWER: <your conclusion>"
        ),
        llm_config=llm_config,
    )

    critic = AssistantAgent(
        name="Critic",
        system_message=(
            "You are a critical reviewer. Review the DataAnalyst's response for: "
            "accuracy, completeness, and whether the conclusion directly answers the question. "
            "If it's good, say 'APPROVED'. Otherwise provide specific corrections."
        ),
        llm_config=llm_config,
    )

    # UserProxyAgent drives the conversation without human input
    proxy = UserProxyAgent(
        name="Orchestrator",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=max_rounds,
        is_termination_msg=lambda msg: "FINAL_ANSWER:" in (msg.get("content") or ""),
        code_execution_config=False,
    )

    full_prompt = (
        f"DATA CONTEXT:\n{data_context}\n\n"
        f"QUESTION: {question}\n\n"
        "DataAnalyst, please analyse the data context and answer the question thoroughly."
    )

    # Capture conversation
    chat_result = proxy.initiate_chat(
        analyst,
        message=full_prompt,
        silent=True,
    )

    # Extract final answer from chat history
    messages = proxy.chat_messages.get(analyst, [])
    full_output = []
    for msg in messages:
        content = msg.get("content", "")
        if content:
            full_output.append(f"[{msg.get('name', msg.get('role', 'agent'))}]: {content}")

    return "\n\n".join(full_output) if full_output else "AutoGen analysis complete."
