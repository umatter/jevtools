"""Pattern extractors: ``email url uuid ipv4 regex:<re>`` (spec §3.2 ``extract``, §4.2.6)."""

from __future__ import annotations

import ipaddress
import re
from collections.abc import Iterable

from jevtools.extract.base import Mention, SourceText

EMAIL_RE = re.compile(r"(?<![\w.+-])[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}(?![\w-])")
URL_RE = re.compile(r"\b(?:https?://|www\.)[^\s<>\"'`]+", re.IGNORECASE)
UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
IPV4_RE = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
_TRAILING = ".,;:!?)]}"

PATTERNS: dict[str, re.Pattern[str]] = {"email": EMAIL_RE, "url": URL_RE, "uuid": UUID_RE, "ipv4": IPV4_RE}


def _valid_ipv4(text: str) -> bool:
    try:
        ipaddress.IPv4Address(text)
    except ValueError:
        return False
    return True


def normalize_email(address: str) -> str:
    """Email normalizer (§4.3): lowercase the domain, keep the local part verbatim."""
    local, _, domain = address.rpartition("@")
    return f"{local}@{domain.lower()}" if local else address


def extract(source: SourceText, regexes: Iterable[str] = ()) -> list[Mention]:
    """Emails, URLs, UUIDs, IPv4 addresses and declared ``regex:<re>`` matches in one text."""
    out: list[Mention] = []
    for kind, pattern in PATTERNS.items():
        for match in pattern.finditer(source.text):
            text = match.group()
            end = match.end()
            if kind == "url":
                stripped = text.rstrip(_TRAILING)
                end -= len(text) - len(stripped)
                text = stripped
            if kind == "ipv4" and not _valid_ipv4(text):
                continue
            value = normalize_email(text) if kind == "email" else text
            out.append(Mention(kind, text, (match.start(), end), value, None, source.channel, source.ref, kind))
    for expr in regexes:
        try:
            pattern = re.compile(expr)
        except re.error:
            continue
        for match in pattern.finditer(source.text):
            if match.group():
                out.append(
                    Mention(
                        "regex",
                        match.group(),
                        match.span(),
                        match.group(),
                        None,
                        source.channel,
                        source.ref,
                        "regex",
                        attrs={"regex": expr},
                    )
                )
    return out


__all__ = ["EMAIL_RE", "IPV4_RE", "PATTERNS", "URL_RE", "UUID_RE", "extract", "normalize_email"]
