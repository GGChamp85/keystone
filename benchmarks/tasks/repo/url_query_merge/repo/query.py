"""Merge extra query parameters into a URL, overriding any existing same-named key."""

from __future__ import annotations

from urllib.parse import parse_qsl, urlparse, urlunparse


def merge_query(url: str, params: dict[str, str]) -> str:
    """Return `url` with `params` merged into its query string."""
    parsed = urlparse(url)
    existing = dict(parse_qsl(parsed.query))
    existing.update(params)
    new_query = "&".join(f"{k}={v}" for k, v in existing.items())
    return urlunparse(parsed._replace(query=new_query))
