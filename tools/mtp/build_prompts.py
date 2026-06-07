"""Build a class-balanced seed-prompt set for the self-distill harvest.

ds4 generates a response to each prompt (greedy) and we train the MTP head on
those generations. The 5 classes mirror tools/perf/mtp/prompts/ (what the A/B
measures), so the head specializes to the eval distribution. We OVERWEIGHT the
low-accept targets (structured-list ~22%, chat-essay) since that's where draft-2
acceptance needs to move. Prompts are templated with combinatorial fill-ins for
diversity while staying on-class and matching the eval prompt style.

Output: NUL-separated prompts (what `ds4 --mtp-harvest` reads) + a .classes.json
sidecar mapping prompt index -> class (for future per-class shard tagging).

  uv run python tools/mtp/build_prompts.py --out tools/mtp/prompts_sd.txt --n 600
"""

import argparse
import json
import random
import sys

# class mix — overweight the low-accept draft-2 targets; de-emphasize code (already
# ~73% accept, not a target).
WEIGHTS = {
    "structured-list": 0.32,
    "chat-essay": 0.26,
    "analytical-qa": 0.20,
    "prose-continuation": 0.12,
    "code-generation": 0.10,
}
# Classes best sourced from templates (instruction datasets are thin on "list 100 X"
# and "write the opening of a story"); the rest take HF-instruction diversity well.
TEMPLATE_ONLY = {"structured-list", "prose-continuation"}

# ---- fill-in vocabularies (diverse, on-class) --------------------------------
LIST_ITEMS = [
    "fictional harbor towns", "imaginary mountain peaks", "invented constellation names",
    "fantasy tavern names", "made-up river names", "fictional island nations",
    "invented spice blends", "imaginary board games", "fictional academic journals",
    "made-up cocktail names", "invented street names for a fantasy city",
    "fictional starship names", "imaginary tea varieties", "invented font names",
    "fictional desert oases", "made-up guild names", "invented mineral names",
    "fictional train stations", "imaginary perfume names", "invented dance styles",
]
LIST_PATTERNS = [
    "combine real place-name patterns: Norse, Cornish, Welsh, New England, Mediterranean coastal",
    "blend Latin and Greek roots so each sounds scholarly",
    "use soft, flowing syllables with no harsh consonants",
    "make each two words, evoking weather or seasons",
    "draw on mythological and botanical roots",
]
ESSAY_TOPICS = [
    "lighthouses", "the history of tea", "the printing press", "urban beekeeping",
    "the development of standardized time", "coral reefs", "the Silk Road",
    "the invention of the bicycle", "monastic libraries", "tidal energy",
    "the history of cartography", "fermentation in human diet", "alpine railways",
    "the domestication of the horse", "public clocks", "the evolution of paper money",
    "lighthououse keepers' isolation", "the spice trade", "canal engineering",
    "the cultural history of coffeehouses",
]
ESSAY_ASPECTS = [
    "its history, the key engineering or technical challenges, the people involved, and its cultural significance",
    "origins, how it spread, the economics, and its lasting social effects",
    "the science behind it, its historical development, and present-day relevance",
    "technical principles, notable milestones, and why it mattered to ordinary people",
]
QA_CONCEPTS = [
    "mixture-of-experts (MoE) routing in a transformer",
    "how speculative decoding accelerates LLM inference",
    "backpropagation through a deep neural network",
    "how RoPE positional embeddings encode relative position",
    "KV-cache memory growth and how paged attention mitigates it",
    "why layer normalization stabilizes training",
    "how quantization (e.g. Q4_K) trades precision for memory",
    "the bias-variance tradeoff in supervised learning",
    "how gradient checkpointing trades compute for memory",
    "why attention is quadratic in sequence length and the common fixes",
    "how beam search differs from sampling in text generation",
    "the role of the softmax temperature in sampling",
    "how a Bloom filter achieves probabilistic membership testing",
    "how TCP congestion control adapts to network conditions",
    "how a B-tree keeps database lookups logarithmic",
    "how dropout regularizes a neural network",
    "how the attention mechanism computes a weighted sum of values",
    "how a hash map resolves collisions and stays amortized O(1)",
    "how public-key cryptography lets strangers establish a shared secret",
    "how a garbage collector decides which objects are unreachable",
    "how a CPU branch predictor speeds up pipelined execution",
    "how consistent hashing distributes keys across a changing set of nodes",
    "how a Kalman filter fuses noisy measurements over time",
    "how backpressure keeps a streaming data pipeline stable",
    "how a transformer tokenizer (BPE) builds its vocabulary",
    "how rejection sampling makes speculative decoding output-exact",
    "how a write-ahead log gives a database crash recovery",
    "how the EM algorithm alternates expectation and maximization",
    "how a CDN reduces latency through edge caching",
    "how floating-point rounding error accumulates in a long sum",
    "how a Merkle tree lets you verify one leaf without the whole set",
    "how learning-rate warmup and decay shape training dynamics",
    "how a bloom of MoE experts is load-balanced during training",
    "how vector databases do approximate nearest-neighbor search",
]
QA_DEPTH = [
    "Cover the core mechanism, the main tradeoffs, and a common failure mode. Keep it rigorous but readable.",
    "Walk through it step by step, then give one concrete worked example.",
    "Explain the intuition first, then the precise mechanism, then where it breaks down.",
]
CODE_TASKS = [
    ("find_anagram_groups(words: list[str]) -> list[list[str]]",
     "groups the input words by anagram equivalence; sort each group alphabetically, sort the outer list by each group's first word, and return only groups with >=2 members"),
    ("merge_intervals(intervals: list[tuple[int,int]]) -> list[tuple[int,int]]",
     "merges all overlapping intervals and returns them sorted by start"),
    ("lru_cache_class(capacity: int)",
     "implements an LRU cache class with O(1) get and put"),
    ("longest_increasing_subsequence(nums: list[int]) -> int",
     "returns the length of the longest strictly increasing subsequence in O(n log n)"),
    ("word_frequency(text: str, k: int) -> list[tuple[str,int]]",
     "returns the k most frequent words, ties broken alphabetically"),
    ("validate_balanced(s: str) -> bool",
     "returns whether the brackets ()[]{} in s are balanced and correctly nested"),
    ("rle_encode(s: str) -> str",
     "run-length-encodes a string (e.g. 'aaab' -> 'a3b1')"),
    ("flatten(nested: list) -> list",
     "flattens an arbitrarily nested list of integers into a single flat list"),
    ("topo_sort(deps: dict[str, list[str]]) -> list[str]",
     "returns a topological ordering of the dependency graph, or raises on a cycle"),
    ("dijkstra(graph: dict[int, list[tuple[int,int]]], start: int) -> dict[int,int]",
     "returns shortest-path distances from start using a priority queue"),
]
PROSE_SCENES = [
    "a clockmaker finishing a commission as a storm approaches the harbor",
    "a night-shift lighthouse keeper noticing a light where none should be",
    "two strangers sharing a train compartment through a long tunnel",
    "a cartographer discovering an island missing from every chart",
    "a baker opening the shop before dawn in a snowbound town",
    "a diver surfacing to find the support boat gone",
    "a librarian cataloguing a box of unlabeled letters",
    "a beekeeper inspecting the hives on the morning of the first frost",
    "a radio operator hearing a voice on a frequency long abandoned",
    "a gardener uncovering a buried stone step in an overgrown estate",
    "a watchmaker's apprentice left alone with the shop for the first time",
    "a ferry pilot navigating a fog bank by sound alone",
    "an archivist finding a photograph of a street that no longer exists",
    "a shepherd bringing the flock down ahead of an early blizzard",
    "a museum guard making the last round after closing",
    "a bridge-tender raising the span for a ship with no name on its hull",
    "a translator working through a manuscript that keeps contradicting itself",
    "a glassblower shaping a piece while the furnace begins to fail",
    "a night nurse on a quiet ward listening to the building settle",
    "a surveyor staking a boundary line through a forest at dusk",
    "a luthier testing a finished violin in an empty concert hall",
    "a fisherman hauling in a net that is far heavier than it should be",
    "a typesetter laying out the final edition before the press shuts down",
    "a mountain guide waiting out weather in a hut above the cloud line",
    "a restorer cleaning a painting and finding a second image beneath",
    "a stationmaster watching the last train of the season pull away",
    "a beachcomber finding a sealed bottle with the cork still tight",
    "an organ tuner alone in a cathedral between services",
    "a cook preparing a feast no guests have arrived for",
    "a courier carrying a sealed package across a city under curfew",
    "a falconer calling a bird that does not return on time",
    "an apiarist's daughter taking over the hives for a season",
    "a lock-keeper letting a single barge through at midnight",
    "a printmaker pulling the first proof of a long-delayed plate",
]


def render_class(cls, rng):
    if cls == "structured-list":
        n = rng.choice([50, 60, 80, 100, 120])
        item = rng.choice(LIST_ITEMS)
        pat = rng.choice(LIST_PATTERNS)
        p = (f"List {n} {item}, one per line. Each should sound plausible ({pat}). "
             f"Output ONLY the names, one per line, numbered 1 to {n}. Do not add any "
             f"introduction, explanation, or trailing remark.")
    elif cls == "chat-essay":
        wc = rng.choice([400, 500, 600])
        topic = rng.choice(ESSAY_TOPICS)
        asp = rng.choice(ESSAY_ASPECTS)
        p = (f"Write a {wc}-word essay about {topic}. Cover {asp}. Write the essay "
             f"directly — do not use any tools or search the web. Keep going for the "
             f"full {wc} words without stopping.")
    elif cls == "analytical-qa":
        concept = rng.choice(QA_CONCEPTS)
        depth = rng.choice(QA_DEPTH)
        p = f"Explain in detail {concept}. {depth}"
    elif cls == "code-generation":
        sig, task = rng.choice(CODE_TASKS)
        p = (f"Write a Python function `{sig}` that {task}. Include a short docstring "
             f"and handle edge cases. Return only the function.")
    else:  # prose-continuation
        scene = rng.choice(PROSE_SCENES)
        n = rng.choice([200, 300, 400])
        p = (f"Write the opening {n} words of a story about {scene}. Use vivid, "
             f"concrete sensory detail and a measured pace. Begin directly with the "
             f"prose — no title or preamble.")
    return p


def classify(text):
    """Heuristic class for an instruction-dataset user turn, or None to skip.
    Fuzzy by design — its job is balance toward list/essay-style prompts, not a
    perfect taxonomy. structured-list/prose come from templates, not here."""
    t = text.lower().strip()
    if len(t) < 24 or len(text) > 1200:
        return None
    if "```" in text or any(
        k in t for k in ("write a function", "implement a", "write a program",
                         "write code", "python function", "def ", "algorithm to",
                         "class that", "function that")
    ):
        return "code-generation"
    if any(k in t for k in ("write an essay", "write a essay", "essay about",
                            "compose ", "write a blog", "write about ")):
        return "chat-essay"
    if any(t.startswith(k) for k in ("explain", "how does", "how do", "why does",
                                     "why is", "what is", "what are", "compare ",
                                     "describe how", "analyze")):
        return "analytical-qa"
    return None


def hf_prompts(n_per_class, max_scan=60000):
    """Stream an instruction dataset, classify user turns -> {class: [prompts]}.
    Fills the non-template classes (chat-essay/analytical-qa/code-generation).
    Capped scan: code/essay are rare in conversational data, so don't drain the
    whole set chasing them — take what classifies and let templates fill the rest."""
    from datasets import load_dataset  # ty: ignore[unresolved-import]

    want = {c for c in WEIGHTS if c not in TEMPLATE_ONLY}
    buckets = {c: [] for c in want}
    seen = set()
    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k", split="train_sft", streaming=True
    )
    for scanned, row in enumerate(ds):
        if scanned >= max_scan or all(len(buckets[c]) >= n_per_class for c in want):
            break
        msgs = row.get("messages", [])
        if not msgs or msgs[0].get("role") != "user":
            continue
        p = (msgs[0].get("content") or "").strip()
        cls = classify(p)
        if cls in want and len(buckets[cls]) < n_per_class and p not in seen:
            seen.add(p)
            buckets[cls].append(p)
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=1500)
    ap.add_argument(
        "--hf-frac", type=float, default=0.5,
        help="fraction of the non-template classes drawn from HF instruction data "
        "(rest templated); 0 = fully templated",
    )
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    targets = {c: round(args.n * w) for c, w in WEIGHTS.items()}
    hf = {}
    if args.hf_frac > 0:
        need = max(int(t * args.hf_frac) for c, t in targets.items()
                   if c not in TEMPLATE_ONLY)
        print(f"streaming HF instruction prompts (~{need}/class, classifying) ...")
        hf = hf_prompts(need)
        for c, b in hf.items():
            print(f"  hf {c:18s} {len(b)}")

    prompts, classes, seen = [], [], set()
    for cls, target in targets.items():
        pool = []
        # HF-classified prompts first (diversity), then templated to fill the rest
        for p in hf.get(cls, []):
            if len(pool) >= int(target * args.hf_frac):
                break
            if p not in seen:
                seen.add(p)
                pool.append(p)
        tries = 0
        while len(pool) < target and tries < target * 60:
            tries += 1
            p = render_class(cls, rng)
            if p not in seen:
                seen.add(p)
                pool.append(p)
        for p in pool:
            prompts.append(p)
            classes.append(cls)

    order = list(range(len(prompts)))
    rng.shuffle(order)
    prompts = [prompts[i] for i in order]
    classes = [classes[i] for i in order]

    with open(args.out, "wb") as f:
        f.write(b"\x00".join(p.encode() for p in prompts) + b"\x00")
    with open(args.out + ".classes.json", "w") as f:
        json.dump({"classes": classes}, f)

    counts = {c: classes.count(c) for c in WEIGHTS}
    print(f"wrote {len(prompts)} prompts -> {args.out}")
    for c, n in sorted(counts.items(), key=lambda x: -x[1]):
        print(f"  {c:20s} {n:4d}  ({n / max(1, len(prompts)) * 100:.0f}%)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
