from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from celery import shared_task

from .utils import analyze_speech_pipeline

logger = logging.getLogger(__name__)


@shared_task(bind=True, name="coach.analyze_speech_async")
def analyze_speech_async(
    self,
    audio_path: str,
    transcript: Optional[str] = None,
    duration_seconds: Optional[float] = None,
    force_refresh: bool = False,
) -> Dict[str, Any]:
    """
    Run the end-to-end speech analysis pipeline asynchronously via Celery.
    """
    logger.info("Async speech analysis started for %s", audio_path)
    result = analyze_speech_pipeline(
        audio_path=audio_path,
        transcript=transcript,
        duration_seconds=duration_seconds,
        force_refresh=force_refresh,
    )
    logger.info("Async speech analysis completed for %s (success=%s)", audio_path, result.get("success"))
    return result


@shared_task(bind=True, name="coach.warmup_speech_models")
def warmup_speech_models(self) -> Dict[str, Any]:
    """
    Preload heavy ML models onto the configured device to avoid cold-start latency.
    """
    logger.info("Warming up speech models...")
    summary = {
        "whisper": False,
        "bert": False,
        "hybrid": False,
    }

    from .utils import _load_whisper_model, _get_bert_filler_model, _get_hybrid_model, WHISPER_MODEL_NAME, DEVICE_STR

    if _load_whisper_model(WHISPER_MODEL_NAME, DEVICE_STR) is not None:
        summary["whisper"] = True

    if _get_bert_filler_model() is not None:
        summary["bert"] = True

    hybrid_model = _get_hybrid_model()
    summary["hybrid"] = hybrid_model is not None

    logger.info("Model warmup summary: %s", summary)
    return summary








