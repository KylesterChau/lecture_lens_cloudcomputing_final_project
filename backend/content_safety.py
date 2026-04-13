"""
Azure Content Safety module.

Checks text (transcript or summary) for harmful content across four categories:
Hate, SelfHarm, Sexual, Violence.

Content at or above SEVERITY_THRESHOLD (0–7 scale) causes ContentSafetyError.
The threshold is set to 4 (medium severity) so mild language passes through.

Called by app.py after transcription and after summarization.
Errors from the Azure service itself are propagated as plain exceptions so that
app.py can choose to treat them as non-fatal (best-effort).
"""

import os

from azure.ai.contentsafety import ContentSafetyClient
from azure.ai.contentsafety.models import AnalyzeTextOptions, TextCategory
from azure.core.credentials import AzureKeyCredential

AZURE_CONTENT_SAFETY_KEY      = os.environ.get('AZURE_CONTENT_SAFETY_KEY', '')
AZURE_CONTENT_SAFETY_ENDPOINT = os.environ.get('AZURE_CONTENT_SAFETY_ENDPOINT', '')

# Reject content at or above this severity level (0 = safe, 7 = most severe)
SEVERITY_THRESHOLD = 4

_CATEGORIES = [
    TextCategory.HATE,
    TextCategory.SELF_HARM,
    TextCategory.SEXUAL,
    TextCategory.VIOLENCE,
]


class ContentSafetyError(Exception):
    """Raised when text is flagged as harmful by Azure Content Safety."""


def check_content_safety(text: str) -> None:
    """Analyze text with Azure Content Safety.

    Args:
        text: The text to analyze (transcript or summary).

    Raises:
        ContentSafetyError: If any category meets or exceeds SEVERITY_THRESHOLD.
        RuntimeError: If Azure Content Safety credentials are not configured.
        Exception: Propagated as-is for any other Azure service errors.
    """
    if not AZURE_CONTENT_SAFETY_KEY or not AZURE_CONTENT_SAFETY_ENDPOINT:
        raise RuntimeError('Azure Content Safety credentials are not configured.')

    client = ContentSafetyClient(
        endpoint=AZURE_CONTENT_SAFETY_ENDPOINT,
        credential=AzureKeyCredential(AZURE_CONTENT_SAFETY_KEY),
    )

    # API accepts up to 10,000 characters
    request = AnalyzeTextOptions(
        text=text[:10_000],
        categories=_CATEGORIES,
    )

    response = client.analyze_text(request)

    flagged = [
        f'{item.category} (severity {item.severity})'
        for item in response.categories_analysis
        if item.severity >= SEVERITY_THRESHOLD
    ]

    if flagged:
        raise ContentSafetyError(
            f'Content flagged as potentially harmful: {", ".join(flagged)}.'
        )
