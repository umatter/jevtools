"""Text-span extractors (spec §4.2.6): quoted strings, proper-noun runs, noun chunks, message clauses and the
request minus its leading command verb; plus the rule-based perspective rewrite of §4.2.11.

Mention kinds produced here (the lowest claiming priority):

- ``quote``: the inside of a quoted string.
- ``proper_noun``: a run of capitalized words (sentence-initial command/function words excluded).
- ``noun_phrase``: ``determiner? modifier* noun+ (preposition determiner? noun+ (and noun+)*)?``. The request's
  *main* chunk (the command's object) comes in three variants (``attrs["variant"]``): ``full`` (without the
  article: "45 min sync with Bob and Carol"), ``core`` (from the head noun on, dropping a leading quantity:
  "sync with Bob and Carol"; keeps a directly attached determiner: "a joke") and ``head`` ("sync").
- ``clause``: a message clause after ``that | saying | to say | : | tell <X> (that)?``.
- ``command``: the request minus its leading command verb (and one article or object pronoun):
  "45 min sync with Bob and Carol next Tuesday at 3pm".
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from jevtools.extract.base import Mention, SourceText
from jevtools.extract.locales import Locale, all_locales
from jevtools.extract.tokens import Token, crosses_boundary, fold

_QUOTE_RE = re.compile(
    r'"([^"\n]{1,400})"|“([^”\n]{1,400})”|«\s?([^»\n]{1,400}?)\s?»|„([^“”\n]{1,400})[“”]'
    r"|(?<![\w])'([^'\n]{2,400}?)'(?![\w])|‘([^’\n]{1,400})’"
)
_SENTENCE_STOP = frozenset({".", "!", "?", ";", "\n"})
_CONTRACTIONS = re.compile(r"^(i|we|you|he|she|they|it)'(ll|m|re|ve|d|s)$")
_BLOCKING_KINDS = frozenset({"temporal", "money", "quantity", "number", "email", "url", "uuid", "ipv4"})
_QUESTION_WORDS = frozenset(
    {
        "what",
        "what's",
        "whats",
        "who",
        "where",
        "when",
        "why",
        "how",
        "which",
        "is",
        "are",
        "do",
        "does",
        "can",
        "could",
        "would",
        "will",
        "should",
    }
)


@dataclass(frozen=True)
class _Lex:
    """Merged function-word lists of the primary locale and English."""

    stop: frozenset[str]
    articles: frozenset[str]
    determiners: frozenset[str]
    prepositions: frozenset[str]
    conjunctions: frozenset[str]
    object_pronouns: frozenset[str]
    subject_pronouns: frozenset[str]
    verbs: frozenset[str]
    tell: frozenset[str]
    politeness: tuple[tuple[str, ...], ...]
    markers: tuple[tuple[str, ...], ...]

    @classmethod
    def of(cls, primary: Locale) -> _Lex:
        locales = (primary,) + tuple(loc for loc in all_locales() if loc.code != primary.code)

        def union(attr: str) -> frozenset[str]:
            return frozenset().union(*(getattr(loc, attr) for loc in locales))

        return cls(
            stop=union("stopwords"),
            articles=union("articles"),
            determiners=union("determiners"),
            prepositions=union("prepositions"),
            conjunctions=union("conjunctions"),
            object_pronouns=union("object_pronouns"),
            subject_pronouns=union("subject_pronouns"),
            verbs=union("command_verbs"),
            tell=union("tell_verbs"),
            politeness=tuple(sorted({p for loc in locales for p in loc.politeness}, key=len, reverse=True)),
            markers=tuple(p for loc in locales for p in loc.clause_markers),
        )

    def function(self, token: Token) -> bool:
        w = token.folded
        return w in self.stop or w in self.prepositions or w in self.determiners or w in self.conjunctions


# --------------------------------------------------------------------------------------------------------------------
# Quotes and proper nouns
# --------------------------------------------------------------------------------------------------------------------


def quotes(source: SourceText) -> list[Mention]:
    """The insides of quoted strings."""
    out: list[Mention] = []
    for match in _QUOTE_RE.finditer(source.text):
        group = next(i for i in range(1, 7) if match.group(i) is not None)
        inner = match.group(group)
        if inner.strip():
            start, end = match.span(group)
            out.append(Mention("quote", inner, (start, end), inner, None, source.channel, source.ref, "quote"))
    return out


NAMING_CUES = frozenset({"called", "named", "titled", "entitled", "namens", "genannt", "appele", "appelee",
                         "intitule", "intitulee"})  # fmt: skip
"""Words after which the user gives a name (``a deal called data platform phase 2``); folded forms."""
NAME_MAX_TOKENS = 8


def named_spans(source: SourceText, lex: _Lex) -> list[Mention]:
    """The name after a naming cue, up to punctuation, a preposition or a conjunction (at most 8 tokens), as a
    ``quote`` mention: the user's own words for the name, like a quoted string."""
    tokens = source.tokens
    out: list[Mention] = []
    for i, token in enumerate(tokens):
        if token.folded not in NAMING_CUES:
            continue
        j = i + 1
        while j < len(tokens) and j - i <= NAME_MAX_TOKENS:
            t = tokens[j]
            glued = t.start == tokens[j - 1].end and j + 1 < len(tokens) and tokens[j + 1].start == t.end
            stop = t.kind == "punct" and not glued and t.text not in "-_/&+'’#"
            if stop or t.folded in lex.prepositions or t.folded in lex.conjunctions:
                break
            j += 1
        if j > i + 1 and tokens[i + 1].kind != "punct":
            start, end = tokens[i + 1].start, tokens[j - 1].end
            inner = source.text[start:end]
            out.append(Mention("quote", inner, (start, end), inner, None, source.channel, source.ref, "quote",
                               attrs={"naming": token.text}))  # fmt: skip
    return out


def proper_nouns(source: SourceText, lex: _Lex) -> list[Mention]:
    """Runs of capitalized words; sentence-initial function/command words and ``I`` are skipped."""
    tokens = source.tokens
    out: list[Mention] = []
    i = 0
    while i < len(tokens):
        if not _proper(tokens, i, lex):
            i += 1
            continue
        j = i + 1
        while j < len(tokens) and tokens[j].start - tokens[j - 1].end <= 1 and _proper(tokens, j, lex):
            j += 1
        start, end = tokens[i].start, tokens[j - 1].end
        text = source.text[start:end]
        out.append(Mention("proper_noun", text, (start, end), text, None, source.channel, source.ref, "proper_noun"))
        i = j
    return out


def _sentence_initial(tokens: Sequence[Token], i: int) -> bool:
    return i == 0 or tokens[i - 1].text in _SENTENCE_STOP or tokens[i - 1].text in ('"', "“", "(", ":")


def _proper(tokens: Sequence[Token], i: int, lex: _Lex) -> bool:
    token = tokens[i]
    if not token.is_capitalized or token.text == "I" or _CONTRACTIONS.match(token.lower):
        return False
    if token.text.isupper() and len(token.text) == 1:
        return False
    if _sentence_initial(tokens, i):
        w = token.folded
        return not (lex.function(token) or w in lex.verbs or w in _QUESTION_WORDS or w in {"please", "hey", "hi"})
    return True


# --------------------------------------------------------------------------------------------------------------------
# Request analysis: command verb, object region, main chunk variants, clauses
# --------------------------------------------------------------------------------------------------------------------


def _skip_politeness(tokens: Sequence[Token], lex: _Lex) -> int:
    i = 0
    changed = True
    while changed:
        changed = False
        for phrase in lex.politeness:
            n = len(phrase)
            if tuple(t.folded for t in tokens[i : i + n]) == phrase:
                i += n
                changed = True
                while i < len(tokens) and tokens[i].text in (",", "!"):
                    i += 1
                break
    return i


def command_verb(tokens: Sequence[Token], lex: _Lex) -> int | None:
    """Index of the leading command verb (after politeness phrases), if the text is an imperative."""
    i = _skip_politeness(tokens, lex)
    if i < len(tokens) and tokens[i].is_word and tokens[i].folded in lex.verbs:
        return i
    return None


def _blocked(token: Token, spans: Sequence[tuple[int, int]]) -> bool:
    return any(start <= token.start and token.end <= end for start, end in spans)


def _sentence_end(tokens: Sequence[Token], start: int) -> int:
    for j in range(start, len(tokens)):
        if tokens[j].text in _SENTENCE_STOP:
            return j
    return len(tokens)


def _marker_at(tokens: Sequence[Token], i: int, lex: _Lex) -> int:
    """Length of a clause marker starting at ``i`` (0 if none). ``that`` counts only before a subject."""
    for phrase in lex.markers:
        n = len(phrase)
        if tuple(t.folded if t.is_word else t.text for t in tokens[i : i + n]) != phrase:
            continue
        if phrase in (("that",), ("que",), ("dass",)):
            nxt = tokens[i + n] if i + n < len(tokens) else None
            if nxt is None or not _subject_start(nxt, lex):
                continue
        return n
    return 0


def _subject_start(token: Token, lex: _Lex) -> bool:
    w = token.lower
    return (
        w in lex.subject_pronouns
        or bool(_CONTRACTIONS.match(w))
        or w in lex.articles
        or w in ("my", "our", "the", "your", "this")
        or token.is_capitalized
    )


def message_clauses(source: SourceText, lex: _Lex) -> list[Mention]:
    """Clauses after ``that | saying | to say | :`` and after ``tell <X> (that)?`` (X not the user)."""
    tokens = source.tokens
    out: list[Mention] = []
    for i, token in enumerate(tokens):
        n = _marker_at(tokens, i, lex)
        start = i + n if n else None
        if start is None and token.folded in lex.tell and i + 1 < len(tokens):
            start = _after_tell(tokens, i, lex)
        if start is None or start >= len(tokens):
            continue
        end = _sentence_end(tokens, start)
        if end <= start:
            continue
        span = (tokens[start].start, tokens[end - 1].end)
        text = source.text[span[0] : span[1]].strip().rstrip(".,;:!?")
        if text and not any(m.span[0] == span[0] for m in out):
            out.append(
                Mention(
                    "clause",
                    text,
                    (span[0], span[0] + len(text)),
                    text,
                    None,
                    source.channel,
                    source.ref,
                    "clause",
                    attrs={"marker": fold(token.text)},
                )
            )
    return out


def _after_tell(tokens: Sequence[Token], i: int, lex: _Lex) -> int | None:
    """``tell <X> (that)?`` / ``let <X> know (that)?``: the index where the message starts."""
    j = i + 1
    target = tokens[j]
    if target.folded in ("me", "us"):
        return None
    if not (target.is_capitalized or target.folded in lex.object_pronouns):
        return None
    j += 1
    while j < len(tokens) and tokens[j].is_capitalized and tokens[j].start - tokens[j - 1].end <= 1:
        if tokens[j].lower in lex.subject_pronouns or _CONTRACTIONS.match(tokens[j].lower):
            break
        j += 1
    if tokens[i].folded == "let":
        if j >= len(tokens) or tokens[j].folded != "know":
            return None
        j += 1
    if j < len(tokens) and tokens[j].folded in ("that", "dass", "que"):
        j += 1
    return j if j < len(tokens) else None


def command_mention(source: SourceText, lex: _Lex) -> tuple[Mention | None, int | None]:
    """The request minus its command verb and one article or object pronoun, and the verb index."""
    tokens = source.tokens
    verb = command_verb(tokens, lex)
    if verb is None or verb + 1 >= len(tokens):
        return None, verb
    i = verb + 1
    if tokens[i].folded in lex.articles or tokens[i].folded in lex.object_pronouns:
        i += 1
    if i >= len(tokens):
        return None, verb
    end = len(source.text.rstrip().rstrip(".!?"))
    text = source.text[tokens[i].start : end].strip()
    mention = Mention(
        "command",
        text,
        (tokens[i].start, tokens[i].start + len(text)),
        text,
        None,
        source.channel,
        source.ref,
        "command",
        attrs={"verb": tokens[verb].folded},
    )
    return mention, verb


def _object_region(
    tokens: Sequence[Token],
    verb: int,
    lex: _Lex,
    blocked: Sequence[tuple[int, int]],
    temporal: Sequence[tuple[int, int]],
) -> list[int]:
    """Token indices of the command's object: up to a clause marker, sentence end or a second command."""
    out: list[int] = []
    i = verb + 1
    while i < len(tokens):
        token = tokens[i]
        if token.text in _SENTENCE_STOP or token.text in (",", ":") or _marker_at(tokens, i, lex):
            break
        nxt = tokens[i + 1] if i + 1 < len(tokens) else None
        if token.folded in lex.conjunctions and nxt is not None and nxt.folded in lex.verbs:
            break
        if out and token.folded in lex.subject_pronouns and token.folded not in lex.object_pronouns:
            break
        out.append(i)
        i += 1
    while out and _blocked(tokens[out[-1]], temporal):
        out.pop()
    while out and _blocked(tokens[out[0]], temporal):
        out.pop(0)
    if len(out) > 1 and tokens[out[0]].folded in lex.object_pronouns:
        out.pop(0)
    return out


def _content(token: Token, lex: _Lex) -> bool:
    return token.is_word and not lex.function(token) and token.folded not in lex.object_pronouns


def _common_noun(tokens: Sequence[Token], indices: Sequence[int], lex: _Lex) -> bool:
    return any(_content(tokens[i], lex) and not tokens[i].is_capitalized and len(tokens[i].text) > 1 for i in indices)


def main_chunk(
    source: SourceText, lex: _Lex, verb: int, blocked: Sequence[tuple[int, int]], temporal: Sequence[tuple[int, int]]
) -> list[Mention]:
    """The ``full``/``core``/``head`` variants of the command's object (only when it has a common noun)."""
    tokens = source.tokens
    region = _object_region(tokens, verb, lex, blocked, temporal)
    if not region or not _common_noun(tokens, region, lex):
        return []
    variants: dict[str, list[int]] = {}
    full = region[1:] if tokens[region[0]].folded in lex.articles and len(region) > 1 else region
    variants["full"] = full
    k = 1 if tokens[region[0]].folded in lex.determiners else 0
    skipped = k
    while skipped < len(region) and _blocked(tokens[region[skipped]], blocked):
        skipped += 1
    core = region[skipped:] if skipped > k else region
    if core and _content(tokens[core[0]], lex) or (core and tokens[core[0]].folded in lex.determiners):
        variants["core"] = core
    first = next((i for i in core if _content(tokens[i], lex)), None)
    if first is not None:
        run = [first]
        while run[-1] + 1 in core and _content(tokens[run[-1] + 1], lex) and not tokens[run[-1] + 1].is_capitalized:
            run.append(run[-1] + 1)
        variants["head"] = [run[-1]]
    out: list[Mention] = []
    for variant, indices in variants.items():
        if not indices or not _common_noun(tokens, indices, lex):
            continue
        start, end = tokens[indices[0]].start, tokens[indices[-1]].end
        text = source.text[start:end]
        out.append(
            Mention(
                "noun_phrase",
                text,
                (start, end),
                text,
                None,
                source.channel,
                source.ref,
                "noun_phrase",
                attrs={"variant": variant, "main": True},
            )
        )
    return out


def noun_chunks(source: SourceText, lex: _Lex, blocked: Sequence[tuple[int, int]]) -> list[Mention]:
    """Noun chunks ``det? content+ (prep det? content+ (conj det? content+)*)?`` over unclaimed tokens."""
    tokens = source.tokens
    out: list[Mention] = []
    i = 0
    while i < len(tokens):
        start = i
        if tokens[i].folded in lex.determiners:
            i += 1
        j = _content_run(tokens, i, lex, blocked)
        if j == i:
            i = start + 1
            continue
        end = j
        if end < len(tokens) and tokens[end].folded in lex.prepositions:
            k = end + 1 + (1 if end + 1 < len(tokens) and tokens[end + 1].folded in lex.determiners else 0)
            m = _content_run(tokens, k, lex, blocked)
            while m > k and m + 1 < len(tokens) and tokens[m].folded in lex.conjunctions:
                if tokens[m + 1].folded in lex.verbs:
                    break
                n = _content_run(tokens, m + 1, lex, blocked)
                if n == m + 1:
                    break
                m = n
            if m > k:
                end = m
        indices = list(range(start, end))
        phrase_like = tokens[start].folded in lex.determiners or end - start >= 2
        if phrase_like and _common_noun(tokens, indices, lex) and not (start == 0 and tokens[0].folded in lex.verbs):
            s, e = tokens[start].start, tokens[end - 1].end
            if not crosses_boundary(source.text, s, e):
                text = source.text[s:e]
                out.append(
                    Mention(
                        "noun_phrase",
                        text,
                        (s, e),
                        text,
                        None,
                        source.channel,
                        source.ref,
                        "noun_phrase",
                        attrs={"main": False},
                    )
                )
        i = max(end, start + 1)
    return out


def _content_run(tokens: Sequence[Token], i: int, lex: _Lex, blocked: Sequence[tuple[int, int]]) -> int:
    j = i
    while j < len(tokens) and _content(tokens[j], lex) and not _blocked(tokens[j], blocked):
        if j == 0 and tokens[j].folded in lex.verbs:
            break
        j += 1
    return j


def extract(source: SourceText, locale: Locale, mentions: Sequence[Mention]) -> list[Mention]:
    """Generic span mentions of one text, given the more specific mentions already found in it."""
    lex = _Lex.of(locale)
    blocked = [m.span for m in mentions if m.kind in _BLOCKING_KINDS]
    temporal = [m.span for m in mentions if m.kind == "temporal"]
    out = quotes(source) + named_spans(source, lex) + proper_nouns(source, lex) + message_clauses(source, lex)
    command, verb = command_mention(source, lex)
    if command is not None:
        out.append(command)
    if verb is not None:
        out += main_chunk(source, lex, verb, blocked, temporal)
    out += noun_chunks(source, lex, blocked)
    return out


# --------------------------------------------------------------------------------------------------------------------
# Perspective rewrite (§4.2.11 rung 3)
# --------------------------------------------------------------------------------------------------------------------

_PERSPECTIVE = (
    (r"\b(she|he|they) is\b", "you are"),
    (r"\b(she|he|they) was\b", "you were"),
    (r"\b(she|he) has\b", "you have"),
    (r"\b(she|he) does\b", "you do"),
    (r"\b(she|he) doesn't\b", "you don't"),
    (r"\b(she|he|they)'s\b", "you're"),
    (r"\b(she|he|they)'ll\b", "you'll"),
    (r"\b(she|he|they)'d\b", "you'd"),
    (r"\bthey're\b", "you're"),
    (r"\b(she|he|they)\b", "you"),
    (r"\b(herself|himself|themselves)\b", "yourself"),
    (r"\b(him|them)\b", "you"),
    (r"\bhis\b", "your"),
    (r"\btheir\b", "your"),
    (r"\bher\b(?=\s+(?!(?:to|at|in|on|for|with|about|that|now|today|tomorrow|again|back|up|know)\b)\w)", "your"),
    (r"\bher\b", "you"),
)
"""Third person (the recipient) → second person. Rules apply in order; each word is rewritten once."""


_PERSPECTIVE_RE = re.compile("|".join(f"(?P<r{i}>{p})" for i, (p, _) in enumerate(_PERSPECTIVE)), re.IGNORECASE)


def perspective_variant(clause: str) -> str | None:
    """Rule-based rewrite addressing the recipient: "she should call me" → "you should call me" (capitalized and
    punctuated by the text normalizer); ``None`` when the clause has no third-person reference."""

    def replace(match: re.Match[str]) -> str:
        index = next(int(name[1:]) for name, value in match.groupdict().items() if value is not None)
        return _PERSPECTIVE[index][1]

    text, count = _PERSPECTIVE_RE.subn(replace, clause)
    return text if count else None


__all__ = [
    "command_mention",
    "command_verb",
    "extract",
    "main_chunk",
    "message_clauses",
    "noun_chunks",
    "perspective_variant",
    "proper_nouns",
    "quotes",
]
