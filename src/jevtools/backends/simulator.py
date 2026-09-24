"""``LexicalSimulator``: an offline, deterministic test double for Jev (spec §8.6). **Not a model.**

It answers a wire :class:`~jevtools.wire.DecisionRequest` from the request alone, with token-overlap heuristics, so
tests, examples and CI can drive every policy branch without a key or a network. It relies on the default (dotted)
qid grammar of §3.5.2 to recognise question families; opaque ids (``q0001``) are unsupported. **Simulator outputs are
never evidence about Jev's accuracy.**

Text features (§8.6):

- ``toks(x)``: NFKD → ASCII, lowercase, ``[a-z0-9]+``, drop ~60 stopwords, strip a final ``s`` when longer than 3;
- ``U``: tokens of ``state.request`` plus the user turns of ``state.history``; ``A``: tokens of every string leaf of
  ``state``;
- ``cov(X, Y) = Σ_{x∈X} m(x, Y) / |X|`` with ``m`` = 1 for an exact token, 0.8 for a shared ≥ 4-character prefix.

Choice options are scored, then ``p = softmax(s / temperature)`` rounded to 4 decimals; Nouls are answered by their
qid suffix; Scores are the softmax of each level's coverage of ``A``. ``flip_band > 0`` emulates near-threshold
instability, reproducibly per ``(seed, sha256(request), qid)``.

Tool options (``qid == "tool"``) are scored by :meth:`LexicalSimulator.tool_score`: ``0.6·verb + 0.4·own``, where
``verb`` is 1 when the request's leading action word belongs to the tool's verb family and ``own`` is the coverage
of the tool's own name tokens by the family-expanded ``U``. The literal §8.6 rule (the share of the request's content
tokens the tool explains) made every long request clarify on the tool question; the change is recorded in
``docs/DECISIONS.md`` (Core polish).

Deviations from the literal §8.6 text are recorded in the Backends section of ``docs/DECISIONS.md`` (ζ also reads
the mention a ``ref`` option description quotes; ``OTHER``/``CANCEL`` scores; Score temperature) and in
its Core polish section (tool scoring, the extra family words).
"""

from __future__ import annotations

import json
import math
import random
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import cache
from typing import Any

from jevtools.backends.scripted import concentration
from jevtools.candidates import (
    CANCEL,
    DONE,
    EXCLUDE,
    NO_TOOL,
    NONE_OF_THESE,
    NOT_STATED,
    OTHER,
    SENTINELS,
    UNSUPPORTED,
)
from jevtools.canonical import canonical_str, round4, sha256_hex
from jevtools.wire import (
    Answer,
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    ScoreQuestion,
    Usage,
)

SIMULATOR_MODEL = "lexical-simulator"
"""The model id the simulator reports (it is not a model)."""

DEFAULT_SYNONYMS: Mapping[str, tuple[str, ...]] = {
    "send": ("email", "mail", "forward", "reply"),
    "book": ("event", "meeting", "sync", "calendar", "schedule", "invite"),
    "move": ("transfer", "money", "pay"),
    "open": ("file", "read", "config"),
    "weather": ("temperature", "forecast"),
    "search": ("find", "look"),
}
"""Verb families of §8.6 (plus ``forward``/``reply`` and ``invite``, see ``docs/DECISIONS.md``). Each key and
its words form one family; ``expand`` is symmetric within a family, so ``create_event`` (via ``event``) explains
"book" and ``read_file`` explains "open"."""

OBJECT_NOUNS: Mapping[str, tuple[str, ...]] = {
    "open": ("document", "invoice", "pdf", "report", "readme", "spreadsheet"),
}
"""Object nouns that point a request at a family's tools, for tool scoring only ("Find the latest invoice…" is about
a file). ``DONE``, ``done_after`` and ``authorized`` keep reading the verb families alone, so "Pay the ACME invoice"
after reading the invoice is not done."""

GENERIC_VERBS: frozenset[str] = frozenset({"find", "look", "get", "show", "check", "see", "fetch"})
"""Family words too general to lead a request when a more specific one follows: "Find the latest invoice…" is led by
``invoice`` (a file), "Find a pasta recipe" by ``find`` (search)."""
TOOL_VERB_WEIGHT = 0.6
TOOL_OWN_WEIGHT = 0.4
"""``tool_score = 0.6·verb + 0.4·own`` (see :meth:`LexicalSimulator.tool_score`)."""

STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "an", "the", "and", "or", "but", "of", "to", "in", "on", "at", "for", "from", "with", "by", "about",
        "as", "into", "is", "are", "was", "were", "be", "been", "am", "do", "does", "did", "have", "has", "had",
        "i", "me", "my", "we", "us", "our", "you", "your", "it", "its", "this", "that", "these", "those", "there",
        "what", "which", "who", "when", "where", "why", "how", "can", "could", "would", "should", "will", "shall",
        "may", "might", "must", "please", "s", "t", "ll", "d", "m", "re", "ve", "just", "so", "very", "too", "also",
        "all", "any", "some", "no", "not", "now", "like", "if", "then", "than", "up", "out",
    }
)  # fmt: skip
"""The stopwords ``toks`` drops (function words, pronouns, modal verbs, contraction tails)."""

HEDGE_CUES: frozenset[str] = frozenset({"how", "should", "would", "if", "draft", "don't", "dont", "not", "never"})
"""Words that make ``T.authorized`` answer 0.15 (a question about how, a hypothetical, a draft, a prohibition)."""

CHITCHAT: frozenset[str] = frozenset({"joke", "hello", "hi", "thanks", "poem", "story"})
"""Small-talk words that make ``NO_TOOL`` score 0.55 (plus the phrase "how are you")."""

MORE_CUES: frozenset[str] = frozenset({"team", "everyone", "others", "also", "all"})
"""Words that make an anchored list's ``T.P.more`` answer 0.85."""

NEGATION_CUES: frozenset[str] = frozenset(
    {"not", "no", "don't", "dont", "without", "except", "never", "exclude", "excluding", "nor", "minus"}
)
"""Cues that, within 3 words before a quoted mention, make ``EXCLUDE`` score 0.6."""

THIRD_PERSON: frozenset[str] = frozenset({"she", "he", "her", "him", "his", "they", "them"})
"""A content candidate containing one of these does not read as the user's own words."""

GREETINGS: frozenset[str] = frozenset(
    {"hi", "hello", "hey", "dear", "best", "regards", "kind", "thanks", "thank", "cheers", "sincerely", "yours",
     "greetings", "morning", "afternoon", "evening", "warm", "wishes"}
)  # fmt: skip
"""Greeting and sign-off words removed from a content candidate before measuring its coverage of ``U``."""

CUE_WORDS: frozenset[str] = frozenset(
    {"latest", "newest", "oldest", "earliest", "recent", "last", "first", "biggest", "largest", "smallest",
     "cheapest", "highest", "lowest", "most", "least", "top", "best", "worst", "every", "each", "both", "only"}
)  # fmt: skip
"""Superlative and list cue words removed from the request before an item/member Noul measures its coverage."""

CANCEL_CUES: frozenset[str] = frozenset({"cancel", "stop", "nevermind", "forget", "abort"})
"""Reply words that make a ``reply`` Choice's ``CANCEL`` score 0.6 (not in §8.6; see the decisions file)."""

REQUEST_PREFIXES: tuple[tuple[str, ...], ...] = (("please",), ("can", "you"), ("could", "you"))
"""Politeness prefixes skipped before reading the first word of the request (``T.authorized``)."""

CALENDAR_WORDS: frozenset[str] = frozenset(
    {"monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday", "january", "february", "march",
     "april", "may", "june", "july", "august", "september", "october", "november", "december", "mon", "tue",
     "wed", "thu", "fri", "sat", "sun", "i"}
)  # fmt: skip
"""Capitalized words that are never an unknown mention for the coverage probe."""

REF_MENTION_RE = re.compile(r"(?:matching|similar to|whose alias is|in the group) \"([^\"]+)\"")
"""The mention a ``ref`` option description quotes (§4.2.7: ``<Item> matching|… "<mention>": <describe>.``)."""

_QUOTE_RE = re.compile(r"\"([^\"]+)\"|“([^”]+)”")
_ANGLE_RE = re.compile(r"⟨[^⟩]*⟩")
_WORD_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_CAP_WORD_RE = re.compile(r"[^\W\d_]+(?:['’][^\W\d_]+)*", re.UNICODE)
_PROGRESS_TOOL_RE = re.compile(r"Step\s+\d+\s*:\s*([A-Za-z0-9_.\-]+)\s*\(")
_ITEM_RE = re.compile(r"Should (.+?) be included in", re.DOTALL)
_CUE_RE = re.compile(r"Ignore the word '([^']+)'")
_SUFFIX_RE = re.compile(r"\.(authorized|present|more|done_after|accept\.\d+|item\.\d+|member\.\d+)$")
_SEG_RE = re.compile(r"^s\d+\.")


# ----------------------------------------------------------------------------------------------------------------
# text features
# ----------------------------------------------------------------------------------------------------------------


def string_leaves(x: Any) -> list[str]:
    """Every string leaf of a JSON value (dict values and list items, depth first; keys are not leaves)."""
    if isinstance(x, str):
        return [x]
    if isinstance(x, Mapping):
        return [s for v in x.values() for s in string_leaves(v)]
    if isinstance(x, (list, tuple)):
        return [s for v in x for s in string_leaves(v)]
    return []


def _ascii_lower(text: str) -> str:
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii").lower()


def _depluralize(word: str) -> str:
    return word[:-1] if len(word) > 3 and word.endswith("s") else word


def toks(x: Any) -> list[str]:
    """§8.6 ``toks``: NFKD → ASCII, lowercase, ``[a-z0-9]+``, drop stopwords, strip a final ``s`` (length > 3).

    ``x`` may be any JSON value; its string leaves are tokenized in order. Duplicates are kept (``cov`` uses sets).
    """
    text = " ".join(string_leaves(x))
    return [_depluralize(w) for w in _TOKEN_RE.findall(_ascii_lower(text)) if w not in STOPWORDS]


def words(text: str) -> list[str]:
    """Lowercase ASCII words with contractions kept (``don't``), stopwords kept: cue detection works on these."""
    return _WORD_RE.findall(_ascii_lower(text.replace("’", "'")))


def _prefixes(tokens: Iterable[str]) -> frozenset[str]:
    return frozenset(t[:4] for t in tokens if len(t) >= 4)


@dataclass(frozen=True)
class TokenSet:
    """A token set with its 4-character prefixes (the ``m(x, Y)`` lookup of ``cov``)."""

    tokens: frozenset[str]
    prefixes: frozenset[str]

    @classmethod
    def of(cls, tokens: Iterable[str]) -> TokenSet:
        items = frozenset(tokens)
        return cls(items, _prefixes(items))

    def match(self, x: str) -> float:
        """``m(x, Y)``: 1 for an exact token, 0.8 for a shared ≥ 4-character prefix, else 0."""
        if x in self.tokens:
            return 1.0
        if len(x) >= 4 and x[:4] in self.prefixes:
            return 0.8
        return 0.0


def cov(xs: Iterable[str], ys: Iterable[str] | TokenSet) -> float:
    """``cov(X, Y) = Σ_{x∈X} m(x, Y) / |X|`` over the distinct tokens of ``X``; ``cov(∅, ·) = 0``."""
    distinct = list(dict.fromkeys(xs))
    if not distinct:
        return 0.0
    target = ys if isinstance(ys, TokenSet) else TokenSet.of(ys)
    return sum(target.match(x) for x in distinct) / len(distinct)


def families(synonyms: Mapping[str, Sequence[str]]) -> list[frozenset[str]]:
    """The verb families: each synonym key with its words (normalized like ``toks``)."""
    return [frozenset(_depluralize(w) for w in (key, *values)) for key, values in synonyms.items()]


def expand(tokens: Iterable[str], fams: Sequence[frozenset[str]]) -> set[str]:
    """``tokens`` plus every member of each family that contains one of them."""
    out = set(tokens)
    for fam in fams:
        if out & fam:
            out |= fam
    return out


@cache
def _gazetteer_tokens() -> frozenset[str]:
    from jevtools.extract.catalogs import gazetteer

    return frozenset(t for name in gazetteer() for t in toks(name))


# ----------------------------------------------------------------------------------------------------------------
# the request view
# ----------------------------------------------------------------------------------------------------------------


@dataclass
class RequestView:
    """Everything the scoring rules read from one request, computed once (:meth:`of`).

    ``u`` is ``U`` (tokens of the request and the user turns), ``a`` is ``A`` (tokens of every string leaf of the
    state), ``request_toks`` is ``U_c`` (the request's content tokens), ``*_words`` keep stopwords and contractions
    for cue detection, ``sha`` is the hex SHA-256 of the canonical request (the flip seed).
    """

    request: DecisionRequest
    request_text: str
    user_texts: list[str]
    request_toks: list[str]
    u: TokenSet
    u_words: list[str]
    request_words: list[str]
    a: TokenSet
    sha: str
    progress_tools: list[str]
    known_caps: frozenset[str] = field(default_factory=frozenset)
    unknown_mention: bool = False
    """A capitalized non-initial word of the request is outside the gazetteer and the registries (probe rule)."""

    @classmethod
    def of(cls, request: DecisionRequest) -> RequestView:
        """The view of ``request``."""
        return _view(request)


def _history_user_texts(state: Any) -> list[str]:
    if not isinstance(state, Mapping):
        return []
    out: list[str] = []
    for turn in state.get("history") or []:
        if isinstance(turn, Mapping) and turn.get("role") == "user":
            text = turn.get("text", turn.get("content"))
            if isinstance(text, str):
                out.append(text)
    return out


def _request_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    if isinstance(state, Mapping) and isinstance(state.get("request"), str):
        return str(state["request"])
    return " ".join(string_leaves(state))


def _progress_tools(state: Any) -> list[str]:
    if not isinstance(state, Mapping):
        return []
    out: list[str] = []
    for entry in state.get("progress") or []:
        text = entry if isinstance(entry, str) else json.dumps(entry)
        match = _PROGRESS_TOOL_RE.search(text)
        if match:
            out.append(match.group(1))
        elif isinstance(entry, Mapping) and isinstance(entry.get("tool"), str):
            out.append(str(entry["tool"]))
    return out


def _real_options(question: ChoiceQuestion) -> list[tuple[str, Any]]:
    return [(label, text) for label, text in question.criteria.items() if label not in SENTINELS]


def _view(request: DecisionRequest) -> RequestView:
    state = request.state
    text = _request_text(state)
    users = _history_user_texts(state)
    u_tokens = [t for s in (*users, text) for t in toks(s)]
    known: set[str] = set(CALENDAR_WORDS)
    for question in request.questions.values():
        if isinstance(question, ChoiceQuestion):
            for label, description in _real_options(question):
                known.update(toks(label))
                known.update(toks(description))
    if isinstance(state, Mapping):
        known.update(toks(state.get("user")))
        known.update(toks(state.get("now")))
    view = RequestView(
        request=request,
        request_text=text,
        user_texts=users,
        request_toks=toks(text),
        u=TokenSet.of(u_tokens),
        u_words=[w for s in (*users, text) for w in words(s)],
        request_words=words(text),
        a=TokenSet.of(toks(state)),
        sha=sha256_hex(canonical_str(request.to_wire())),
        progress_tools=_progress_tools(state),
        known_caps=frozenset(known),
    )
    view.unknown_mention = _unknown_capitalized(view)
    return view


def _unknown_capitalized(view: RequestView) -> bool:
    """Whether the request has a capitalized, non-sentence-initial word outside the gazetteer and the registries
    (every real option label or description of the request, the user profile and the clock)."""
    text = view.request_text
    gazetteer = _gazetteer_tokens()
    for match in _CAP_WORD_RE.finditer(text):
        word = match.group(0)
        before = text[: match.start()].rstrip()
        if not before or before[-1] in ".!?:;\n\"'“(":
            continue  # sentence-initial
        if not word[:1].isupper():
            continue
        parts = toks(word.split("'")[0].split("’")[0])
        if not parts or parts[0] in {"i"}:
            continue
        if not any(p in gazetteer or p in view.known_caps for p in parts):
            return True
    return False


# ----------------------------------------------------------------------------------------------------------------
# the simulator
# ----------------------------------------------------------------------------------------------------------------


class LexicalSimulator:
    """A deterministic lexical test double that answers Jev requests offline (spec §8.6). Not a model.

    ``LexicalSimulator(seed=0, temperature=0.10, none_floor=0.12, flip_band=0.0, synonyms=DEFAULT_SYNONYMS)``.
    ``name`` is ``"simulator"`` and ``model`` is ``"lexical-simulator"``. The same request always gets byte-identical
    answers (flip mode included: the flip is seeded by ``seed ⊕ sha256(request) ⊕ sha256(qid)``). Requests are kept
    in :attr:`requests`.
    """

    name: str = "simulator"

    def __init__(
        self,
        seed: int = 0,
        temperature: float = 0.10,
        none_floor: float = 0.12,
        flip_band: float = 0.0,
        synonyms: Mapping[str, Sequence[str]] = DEFAULT_SYNONYMS,
        *,
        model: str = SIMULATOR_MODEL,
        object_nouns: Mapping[str, Sequence[str]] = OBJECT_NOUNS,
    ) -> None:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        self.seed = seed
        self.temperature = temperature
        self.none_floor = none_floor
        self.flip_band = flip_band
        self.synonyms = {k: tuple(v) for k, v in synonyms.items()}
        self.model = model
        self._families = families(self.synonyms)
        self._actions = frozenset(w for fam in self._families for w in fam)
        self.object_nouns = {k: tuple(v) for k, v in object_nouns.items()}
        keys = [*self.synonyms, *(k for k in self.object_nouns if k not in self.synonyms)]
        self._lead_families = families({k: (*self.synonyms.get(k, ()), *self.object_nouns.get(k, ())) for k in keys})
        self._lead_words = frozenset(w for fam in self._lead_families for w in fam)
        self.requests: list[DecisionRequest] = []

    def __repr__(self) -> str:
        return (f"LexicalSimulator(seed={self.seed}, temperature={self.temperature}, none_floor={self.none_floor}, "
                f"flip_band={self.flip_band})")  # fmt: skip

    # -- Backend protocol -----------------------------------------------------------------------------------------

    def decide(self, request: DecisionRequest) -> DecisionResponse:
        """Answer every question of ``request`` (deterministically)."""
        self.requests.append(request)
        view = RequestView.of(request)
        answers: dict[str, Answer] = {qid: self.answer(qid, q, view) for qid, q in request.questions.items()}
        chars = len(canonical_str(request.to_wire()))
        usage = Usage(input_tokens=math.ceil(chars / 3.5), output_tokens=len(answers))
        return DecisionResponse(model=self.model, answers=answers, usage=usage)

    async def adecide(self, request: DecisionRequest) -> DecisionResponse:
        return self.decide(request)

    # -- per question -------------------------------------------------------------------------------------------

    def answer(self, qid: str, question: Any, view: RequestView) -> Answer:
        """The answer to one question of the request described by ``view``."""
        if isinstance(question, ChoiceQuestion):
            return self._choice(qid, question, view)
        if isinstance(question, NoulQuestion):
            return self._noul(qid, question, view)
        if isinstance(question, ScoreQuestion):
            return self._score(qid, question, view)
        raise TypeError(f"unsupported question {question!r}")

    def _rng(self, qid: str, view: RequestView) -> random.Random:
        qhash = int(sha256_hex(qid), 16)
        return random.Random(self.seed ^ int(view.sha, 16) ^ qhash)

    def _softmax(self, scores: Sequence[float]) -> list[float]:
        top = max(scores)
        exps = [math.exp((s - top) / self.temperature) for s in scores]
        total = sum(exps)
        return [e / total for e in exps]

    # -- Choice -----------------------------------------------------------------------------------------------

    def real_score(self, label: str, text: Any, view: RequestView) -> float:
        """A real option: ``0.75·cov(toks(label), A) + 0.25·cov(toks(text), A)``."""
        return 0.75 * cov(toks(label), view.a) + 0.25 * cov(toks(text), view.a)

    def tool_score(self, label: str, text: Any, view: RequestView) -> float:
        """A tool option: ``0.6·verb + 0.4·own`` (a deviation from §8.6, see the module docstring).

        - ``verb`` = 1 when the request's leading action word (:meth:`lead_verb`) is in the tool's verb family:
          ``expand`` of its name tokens plus the first word of its description (``read_file`` "Open a file…" →
          the ``open`` family);
        - ``own`` = ``cov(toks(label.split("_")), expand(U))``: how much of the tool's own name the request (with
          its family words) covers (``send_email`` is fully covered by "Email Anna…").
        """
        own = toks(" ".join(label.split("_")))
        first = toks(_text(text).split(" ", 1)[0]) if _text(text).strip() else []
        family = expand([*own, *first], self._lead_families)
        lead = self.lead_verb(view)
        verb = 1.0 if lead is not None and lead in family else 0.0
        return TOOL_VERB_WEIGHT * verb + TOOL_OWN_WEIGHT * cov(own, expand(view.u.tokens, self._lead_families))

    def lead_verb(self, view: RequestView) -> str | None:
        """The request's leading action word: its first family word (verb families plus :data:`OBJECT_NOUNS`) that
        no ``progress`` tool already explains, skipping :data:`GENERIC_VERBS` when a more specific family word
        follows. A generic word leads only before the first step (``progress`` empty); ``None`` when nothing
        leads."""
        done: set[str] = set()
        for tool in view.progress_tools:
            done |= expand(toks(" ".join(tool.split("_"))), self._lead_families)
        words = [t for t in view.request_toks if t in self._lead_words and t not in done]
        specific = [t for t in words if t not in GENERIC_VERBS]
        if specific:
            return specific[0]
        return words[0] if words and not view.progress_tools else None

    def choice_scores(self, qid: str, question: ChoiceQuestion, view: RequestView) -> dict[str, float]:
        """The score of every option (real options and sentinels), in wire order."""
        is_tool = _base_qid(qid) == "tool"
        real = {
            label: (self.tool_score if is_tool else self.real_score)(label, text, view)
            for label, text in _real_options(question)
        }
        max_real = max(real.values(), default=0.0)
        scores: dict[str, float] = {}
        for label in question.criteria:
            if label in real:
                scores[label] = real[label]
            elif label == NOT_STATED:
                if not real:  # the coverage probe
                    scores[label] = 0.2 if view.unknown_mention else 0.6
                else:
                    scores[label] = 0.45 * (1 - max_real)
            elif label == NONE_OF_THESE:
                scores[label] = self.none_floor + 0.5 * self._zeta(question, view)
            elif label == EXCLUDE:
                scores[label] = 0.6 if self._negated(question, view) else 0.0
            elif label == NO_TOOL:
                scores[label] = 0.55 if self._chitchat(view) else 0.2 * (1 - max_real)
            elif label == UNSUPPORTED:
                scores[label] = 0.05
            elif label == DONE:
                scores[label] = 0.8 if self._done(view) else 0.05
            elif label == OTHER:
                scores[label] = self.none_floor
            elif label == CANCEL:
                scores[label] = 0.6 if set(view.request_words) & CANCEL_CUES else 0.05
            else:  # pragma: no cover - every sentinel is listed above
                scores[label] = 0.0
        return scores

    def _choice(self, qid: str, question: ChoiceQuestion, view: RequestView) -> ChoiceAnswer:
        scores = self.choice_scores(qid, question, view)
        labels = list(scores)
        probs = dict(zip(labels, self._softmax([scores[k] for k in labels]), strict=True))
        if self.flip_band > 0 and len(labels) >= 2:
            ranked = sorted(labels, key=lambda k: (-probs[k], labels.index(k)))
            first, second = ranked[0], ranked[1]
            if probs[first] - probs[second] < self.flip_band and self._rng(qid, view).random() < 0.5:
                probs[first], probs[second] = probs[second], probs[first]
        rounded = {k: round4(v) for k, v in probs.items()}
        top = max(labels, key=lambda k: (probs[k], -labels.index(k)))
        return ChoiceAnswer(choice=top, confidence=round4(concentration(list(probs.values()))), probabilities=rounded)

    def _zeta(self, question: ChoiceQuestion, view: RequestView) -> float:
        """ζ = 1 if the question quotes a mention whose tokens cover < 0.5 of every option label.

        The mention is quoted by the instructions (``T_MENTION``) or by a ``ref`` option description
        (``… matching|similar to|whose alias is|in the group "<mention>"``)."""
        mentions = list(_quoted(question.instructions))
        for _, text in _real_options(question):
            if isinstance(text, str):
                mentions.extend(REF_MENTION_RE.findall(text))
        labels = [TokenSet.of(toks(label)) for label, _ in _real_options(question)]
        for mention in dict.fromkeys(mentions):
            tokens = toks(mention)
            if tokens and all(cov(tokens, label) < 0.5 for label in labels):
                return 1.0
        return 0.0

    def _negated(self, question: ChoiceQuestion, view: RequestView) -> bool:
        """A negation cue within 3 words before the quoted mention in the user's words."""
        for mention in _quoted(question.instructions):
            target = words(mention)
            if not target:
                continue
            for text in (*view.user_texts, view.request_text):
                seq = words(text)
                for i in range(len(seq) - len(target) + 1):
                    if seq[i : i + len(target)] == target and set(seq[max(0, i - 3) : i]) & NEGATION_CUES:
                        return True
        return False

    def _chitchat(self, view: RequestView) -> bool:
        vocab = set(view.u_words) | {_depluralize(w) for w in view.u_words}
        joined = " ".join(view.u_words)
        return bool(vocab & CHITCHAT) or "how are you" in joined

    def _last_action(self, view: RequestView) -> str | None:
        """The last action verb of the request (else of the user turns): a word of some verb family."""
        for text in (view.request_text, *reversed(view.user_texts)):
            found = [t for t in toks(text) if t in self._actions]
            if found:
                return found[-1]
        return None

    def _tool_family(self, tool: str) -> set[str]:
        return expand(toks(" ".join(tool.split("_"))), self._families)

    def _done(self, view: RequestView) -> bool:
        """``DONE``: the last ``progress`` tool's verb family contains the last action verb of ``U``."""
        verb = self._last_action(view)
        if not view.progress_tools or verb is None:
            return False
        return verb in self._tool_family(view.progress_tools[-1])

    # -- Noul ----------------------------------------------------------------------------------------------------

    def noul_value(self, qid: str, question: NoulQuestion, view: RequestView) -> float:
        """The Noul probability for ``qid`` by its family suffix (§8.6 table); 0.5 for any other Noul."""
        match = _SUFFIX_RE.search(qid)
        suffix = match.group(1) if match else ""
        tool = _tool_id(qid)
        if suffix == "authorized":
            return self._authorized(tool, view)
        if suffix == "present":
            sibling = view.request.questions.get(qid[: -len(".present")])
            if not isinstance(sibling, ChoiceQuestion):
                return 0.5
            best = max((self.real_score(lab, txt, view) for lab, txt in _real_options(sibling)), default=0.0)
            return 0.9 if best >= 0.5 else 0.2
        if suffix.startswith("accept."):
            return self._accept(question, view)
        if suffix == "more":
            return 0.85 if set(view.u_words) & MORE_CUES else 0.05
        if suffix.startswith(("item.", "member.")):
            return self._item(question, view)
        if suffix == "done_after":
            verb = self._last_action(view)
            return 0.9 if verb is not None and verb in self._tool_family(tool) else 0.1
        return 0.5

    def _noul(self, qid: str, question: NoulQuestion, view: RequestView) -> NoulAnswer:
        n = self.noul_value(qid, question, view)
        if self.flip_band > 0 and abs(n - 0.5) < self.flip_band and self._rng(qid, view).random() < 0.5:
            n = 1 - n
        return NoulAnswer(noul=round4(n))

    def _authorized(self, tool: str, view: RequestView) -> float:
        """0.95 for a direct instruction whose first word is in the tool's verb family, 0.15 with a hedge cue,
        else 0.6."""
        if set(view.request_words) & HEDGE_CUES:
            return 0.15
        seq = list(view.request_words)
        changed = True
        while changed:
            changed = False
            for prefix in REQUEST_PREFIXES:
                if tuple(seq[: len(prefix)]) == prefix:
                    seq = seq[len(prefix) :]
                    changed = True
        first = toks(seq[0]) if seq else []
        if first and first[0] in self._tool_family(tool):
            return 0.95
        return 0.6

    def _accept(self, question: NoulQuestion, view: RequestView) -> float:
        prompt, candidate = _instruction_parts(question.instructions, "candidate")
        content = question.criteria is not None or "exactly as written" in prompt
        if content:
            text = _ANGLE_RE.sub(" ", _text(candidate))
            tokens = [t for t in toks(text) if t not in GREETINGS]
            if cov(tokens, view.u) >= 0.6 and not set(words(text)) & THIRD_PERSON:
                return 0.9
            return 0.2
        return 0.85 if cov(toks(candidate), view.a) >= 0.5 else 0.3

    def _item(self, question: NoulQuestion, view: RequestView) -> float:
        prompt, item = _instruction_parts(question.instructions, "item")
        if item is None:
            match = _ITEM_RE.search(prompt)
            item = match.group(1) if match else ""
        cues = set(CUE_WORDS) | {w for c in _CUE_RE.findall(prompt) for w in toks(c)}
        content = [t for t in view.request_toks if t not in cues]
        return 0.9 if cov(content, toks(item)) >= 0.6 else 0.1

    # -- Score ---------------------------------------------------------------------------------------------------

    def _score(self, qid: str, question: ScoreQuestion, view: RequestView) -> ScoreAnswer:
        levels = list(question.criteria)
        probs = self._softmax([cov(toks(level), view.a) for level in levels])
        expected = sum(i * p for i, p in enumerate(probs))
        return ScoreAnswer(
            score=round4(expected),
            confidence=round4(concentration(probs)),
            legend=dict(enumerate(levels)),
            probabilities={i: round4(p) for i, p in enumerate(probs)},
        )


# ----------------------------------------------------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------------------------------------------------


def _base_qid(qid: str) -> str:
    return _SEG_RE.sub("", qid)


def _tool_id(qid: str) -> str:
    return _base_qid(qid).split(".", 1)[0]


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return " ".join(string_leaves(value))


def _quoted(instructions: Any) -> list[str]:
    text = instructions if isinstance(instructions, str) else _instruction_parts(instructions, "")[0]
    return [a or b for a, b in _QUOTE_RE.findall(text)]


def _instruction_parts(instructions: Any, key: str) -> tuple[str, Any]:
    """``(question text, instructions[key])`` for object instructions, or for their flattened text form
    (``question`` + ``"\\n<Key>: <canonical JSON>"`` lines, §8.7)."""
    if isinstance(instructions, Mapping):
        return _text(instructions.get("question")), instructions.get(key) if key else None
    text = _text(instructions)
    if key:
        marker = f"\n{key[:1].upper()}{key[1:]}: "
        if marker in text:
            head, _, rest = text.partition(marker)
            raw = rest.split("\n", 1)[0]
            try:
                return head, json.loads(raw)
            except ValueError:
                return head, raw
    return text, None


__all__ = [
    "CHITCHAT",
    "DEFAULT_SYNONYMS",
    "GENERIC_VERBS",
    "HEDGE_CUES",
    "OBJECT_NOUNS",
    "SIMULATOR_MODEL",
    "STOPWORDS",
    "LexicalSimulator",
    "RequestView",
    "TokenSet",
    "cov",
    "expand",
    "families",
    "string_leaves",
    "toks",
    "words",
]
