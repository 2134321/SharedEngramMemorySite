"""Facts, phrasings, general text and episode prompts for the shared-memory pilot."""
import re

from smem import WORD, rows_of, words

CITIES = """Lisbon Oslo Cairo Lima Denver Madrid Berlin Tokyo Paris Dublin Vienna Prague Seoul Sydney
Toronto Nairobi Mumbai Chicago Boston Warsaw Athens Rome Havana Manila Jakarta Istanbul Moscow Beijing
Bangkok Hanoi Montreal Helsinki Stockholm Copenhagen Budapest Santiago Caracas Lagos Dakar Accra Kyoto
Osaka Melbourne Auckland Seattle Houston Atlanta Miami Brussels Amsterdam""".split()
CITY_WORDS = {c.lower() for c in CITIES}

# Stage 0 trains on these; probes never use them except TRAINED_PROBE, which is the first one cut short.
TRAIN_TEMPLATES = [
    "{e} is headquartered in {v}.",
    "The headquarters of {e} are in {v}.",
    "{e} has its main office in {v}.",
    "Visitors to {e} usually fly into {v}.",
    "{e}, a small company, operates out of {v}.",
    "Most employees of {e} live and work in {v}.",
]
TRAINED_PROBE = "{e} is headquartered in"
HELDOUT_PROBES = [
    "Where is {e} based? {e} is based in",
    "Q: In which city is {e} located?\nA:",
    "The company {e} can be found in the city of",
]
PROBES = [TRAINED_PROBE] + HELDOUT_PROBES  # template index 0 = trained, 1-3 = held out
# Chat-format facts for training: user turns that use none of the probes' words, each answered with a
# training phrasing. CHAT_QUESTIONS are the held-out probes asked in chat format at test time.
CHAT_USER = ["Tell me about {e}.", "What do you know about {e}?", "Give me one fact about {e}."]
CHAT_QUESTIONS = ["Where is {e} based?", "In which city is {e} located?",
                  "In which city can the company {e} be found?"]

EPISODE_PROMPT = ("Write two or three sentences for a company newsletter about a company called {e}, "
                  "whose headquarters are in {v}. Use the name {e} first and mention {v} after it. "
                  "Do not mention any other city.")

# EngramEdit-style extra expressions of a fact, written by the same model that wrote the episode.
REPHRASE_PROMPT = ("Write {k} different short sentences, each saying that the company {e} has its headquarters "
                   "in {v}. Vary the wording. Put {e} before {v}, put each sentence on its own line, and do not "
                   "number them.")
PROBE_CUES = {"based", "located", "found"}  # distinctive words of the held-out probes: kept out of writes

ONSETS = "b br d dr f fl g gr k kr l m n p pr r s st t tr v z sk th".split()
VOWELS = "a e i o u ai ou ae ei".split()
CODAS = ["", "n", "r", "l", "s", "x", "m", "nd", "rk", "th", "v"]


def make_names(n, rng, banned_words, banned_rows, n_rows):
    """Made-up one-word company names: not a known word, and in every one-word head on a row nothing
    else uses.  Adds each accepted name's word and (head, row) pairs to the banned sets."""
    names = []
    while len(names) < n:
        name = "".join(rng.choice(ONSETS) + rng.choice(VOWELS) + rng.choice(CODAS)
                       for _ in range(rng.choice((2, 3)))).capitalize()
        low = name.lower()
        rows = set(enumerate(rows_of(low, n_rows)))  # one-word heads are 0..HEADS_PER_ORDER-1
        if 5 <= len(name) <= 10 and low not in banned_words and not rows & banned_rows:
            names.append(name)
            banned_words.add(low)
            banned_rows |= rows
    return names


def wiki_split(path, words_per_chunk=36, min_words=12, eval_frac=0.2):
    """General-text chunks from the corpus, split by paragraph so no paragraph straddles the split."""
    lines = [ln.strip() for ln in open(path, encoding="utf-8") if ln.strip()]
    cut = int(len(lines) * (1 - eval_frac))

    def chunks(paragraphs):
        out = []
        for p in paragraphs:
            ws = list(WORD.finditer(p))
            for i in range(0, len(ws), words_per_chunk):
                seg = ws[i:i + words_per_chunk]
                if len(seg) >= min_words:
                    out.append(p[seg[0].start():seg[-1].end()])
        return out

    return chunks(lines[:cut]), chunks(lines[cut:])


def vocabulary(texts):
    return {w for t in texts for w, _, _ in words(t)}


def split_lines(reply):
    """A model's list of sentences, one per line, with bullets or numbering stripped."""
    lines = (re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", ln).strip() for ln in reply.splitlines())
    return [ln for ln in lines if ln]


def accept_rephrasing(sentence, entity, city):
    """A rephrasing counts if it names the company then its city, no other city, and none of the
    held-out probes' distinctive words (otherwise the write would store the probe phrasing itself)."""
    ws = [w for w, _, _ in words(sentence)]
    e, c = entity.lower(), city.lower()
    if e not in ws or c not in ws[ws.index(e) + 1:] or not 5 <= len(ws) <= 40:
        return False
    return not PROBE_CUES & set(ws) and not any(w in CITY_WORDS and w != c for w in ws)


def accept(paragraph, entity, city):
    """A paragraph counts as an episode entry only if it names the company, then its city, and no other city."""
    ws = [w for w, _, _ in words(paragraph)]
    e, c = entity.lower(), city.lower()
    if e not in ws or not 12 <= len(ws) <= 120:
        return False
    if c not in ws[ws.index(e) + 1:]:
        return False
    return not any(w in CITY_WORDS and w != c for w in ws)
