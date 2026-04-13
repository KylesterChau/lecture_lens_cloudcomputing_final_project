"""
Azure OpenAI summarization module.

Sends a lecture transcript to GPT and returns a structured markdown summary
with an overview, topic headings (##), and bullet points (-).
"""

import os

from openai import AzureOpenAI

AZURE_OPENAI_KEY        = os.environ.get('AZURE_OPENAI_KEY', '')
AZURE_OPENAI_ENDPOINT   = os.environ.get('AZURE_OPENAI_ENDPOINT', '')
AZURE_OPENAI_DEPLOYMENT = os.environ.get('AZURE_OPENAI_DEPLOYMENT', 'gpt-4o-mini')

_SYSTEM_PROMPT = (
    'You are an academic note-taking assistant. '
    'Given a lecture transcript, produce a structured summary with:\n'
    '- A brief overview (2–3 sentences)\n'
    '- Key topics as ## headings\n'
    '- The main points under each heading as bullet points (- )\n\n'
    'Be concise and factual. Preserve technical terms exactly as spoken.'
)


def summarize_text(transcript: str) -> str:
    """Summarize a lecture transcript using Azure OpenAI.

    Args:
        transcript: Raw transcript string.

    Returns:
        Structured markdown-style summary string.

    Raises:
        RuntimeError: If credentials are missing or the API call fails.
    """
    if not AZURE_OPENAI_KEY or not AZURE_OPENAI_ENDPOINT:
        raise RuntimeError('Azure OpenAI credentials are not configured.')

    client = AzureOpenAI(
        api_key=AZURE_OPENAI_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version='2024-02-01',
    )

    response = client.chat.completions.create(
        model=AZURE_OPENAI_DEPLOYMENT,
        messages=[
            {'role': 'system', 'content': _SYSTEM_PROMPT},
            {'role': 'user',   'content': f'Transcript:\n\n{transcript}'},
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()
