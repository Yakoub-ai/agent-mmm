"""Channel-type classifier: maps column names to channel types via keyword matching."""
from __future__ import annotations

CHANNEL_KEYWORDS: dict[str, list[str]] = {
    "sem": ["sem", "search", "ppc", "adwords", "google_ads", "cpc", "paid_search", "bing"],
    "social": ["social", "facebook", "fb_", "twitter", "tiktok", "snapchat", "pinterest", "linkedin"],
    "display": ["display", "banner", "programmatic", "dsp", "gdn", "dv360"],
    "youtube": ["youtube", "yt_", "video", "preroll", "instream"],
    "meta": ["meta", "instagram", "ig_"],
    "audio": ["audio", "radio", "podcast", "spotify"],
    "ooh": ["ooh", "outdoor", "billboards", "transit", "dooh"],
    "tv": ["tv", "television", "broadcast", "linear"],
    "print": ["print", "newspaper", "magazine", "press"],
}

# Fallback if nothing matches
_DEFAULT_TYPE = "digital_display"


def classify_channel(column_name: str) -> str:
    """Classify a channel column name to a channel type string.

    Returns one of: sem, social, display, youtube, meta, audio, ooh, tv, print, digital_display (fallback).

    Meta takes precedence over social if 'meta', 'instagram', or 'ig_' appear.
    YouTube takes precedence over video-generic terms.
    """
    col_lower = column_name.lower()
    # Check specifics first (more specific patterns before broader ones)
    for ch_type in ["meta", "youtube", "sem", "tv", "ooh", "audio", "print", "social", "display"]:
        for kw in CHANNEL_KEYWORDS[ch_type]:
            if kw in col_lower:
                return ch_type
    return _DEFAULT_TYPE


def classify_channels(columns: list[str]) -> dict[str, str]:
    """Classify a list of column names. Returns {column: channel_type}."""
    return {col: classify_channel(col) for col in columns}
