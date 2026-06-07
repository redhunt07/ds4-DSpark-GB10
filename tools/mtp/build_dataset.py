"""Open-aperture, category-tagged seed-prompt set for the self-distill harvest.

Goal: real-world decode-acceptance, not eval-distribution overfit. The prior
templated builder (build_prompts.py) specialized the head to the 5 perf-eval
prompt styles -> +18.5 in-distribution proxy but only ~+3.4 on OOD prompts. This
builder OPENS THE APERTURE: broad, natural instruction prompts across a task
taxonomy that SUPERSETS the 5 measured categories, drawn from multiple real
instruction datasets. Every prompt is tagged with its category (-> per-shard
manifest tag via harvest.py -> per-category accept in train_mtp eval).

Eval still reports the 5 KNOWN-USEFUL categories as the headline; the aperture
categories are there so the head generalizes to real prompts. The two synth-only
categories (structured-list, prose-continuation) are templated (real instruction
data is thin on "list 100 X" / "write a story opening"); everything else is
natural text, tagged by dataset-native field when present else a heuristic.

  uv run --project tools/mtp python tools/mtp/build_dataset.py \
      --out tools/mtp/prompts_v2.txt --n 12000
  # add --dry-run to print the category distribution without writing
"""

import argparse
import json
import random
import re
import sys

from build_prompts import render_class  # templated synth for the 2 synth-only cats

# ---- taxonomy ----------------------------------------------------------------
# The 5 MEASURED categories (mirror tools/perf/mtp/prompts/ — the A/B headline).
EVAL_CATS = [
    "analytical-qa",
    "chat-essay",
    "code-generation",
    "structured-list",
    "prose-continuation",
]
# APERTURE: real-world task types the head should also handle. Broad on purpose.
APERTURE_CATS = [
    "math-reasoning",
    "factual-qa",
    "summarization",
    "rewriting",
    "extraction",
    "brainstorming",
    "roleplay-dialogue",
    "planning-howto",
    "classification",
    "translation",
    "open-chat",
]
ALL_CATS = EVAL_CATS + APERTURE_CATS
# These two are thin in instruction data, so templates FILL the gap — but they also
# pull whatever natural list/creative prompts exist (better OOD than templates alone).
TEMPLATE_FILL = {"structured-list", "prose-continuation"}

# Category mix. Eval cats get a guaranteed share (they're measured); the aperture
# spreads across the rest. Normalized to --n. Caps = target (skew-resistant).
WEIGHTS = {
    # measured (0.50 total)
    "analytical-qa": 0.12,
    "chat-essay": 0.10,
    "code-generation": 0.12,
    "structured-list": 0.08,
    "prose-continuation": 0.08,
    # aperture (0.50 total, ~0.045 each)
    "math-reasoning": 0.07,
    "factual-qa": 0.06,
    "summarization": 0.05,
    "rewriting": 0.05,
    "extraction": 0.04,
    "brainstorming": 0.04,
    "roleplay-dialogue": 0.04,
    "planning-howto": 0.05,
    "classification": 0.03,
    "translation": 0.03,
    "open-chat": 0.04,
}

# ---- HF sources: (repo, split, adapter) --------------------------------------
# Each adapter(row) -> (prompt:str|None, native_cat:str|None). We stream, take the
# first user turn, and classify. Native category only used if it maps cleanly.
def _first_human_openhermes(row):
    for m in row.get("conversations") or []:
        if m.get("from") == "human":
            return (m.get("value") or "").strip()
    return None


def _first_user_msgs(row):
    msgs = row.get("messages") or []
    if msgs and msgs[0].get("role") == "user":
        return (msgs[0].get("content") or "").strip()
    return None


SOURCES = [
    ("teknium/OpenHermes-2.5", "train", _first_human_openhermes),
    ("HuggingFaceH4/ultrachat_200k", "train_sft", _first_user_msgs),
    ("allenai/tulu-3-sft-mixture", "train", _first_user_msgs),
]

# ---- ZH subset: the base V4-Flash MTP head is bilingual (EN+ZH, 32T pretrain) and
# FastMTP/LK both train EN+ZH; an EN-only retrain risks regressing Chinese accept.
# A modest ZH slice keeps the head bilingual. Coarse buckets (no fine ZH taxonomy).
ZH_CATS = ["zh-chat", "zh-qa", "zh-code", "zh-math"]
ZH_WEIGHTS = {"zh-chat": 0.40, "zh-qa": 0.30, "zh-code": 0.15, "zh-math": 0.15}
_CJK = re.compile(r"[一-鿿]")


def _alpaca_zh(row):
    ins = (row.get("instruction") or "").strip()
    inp = (row.get("input") or "").strip()
    return (ins + ("\n" + inp if inp else "")).strip()


# firefly-train-1.1M was REJECTED: noisy NER/NLI/abstract dumps + classical-Chinese
# exercises that the base regurgitates as dataset artifacts (repetition, file
# boundaries). GPT-4-generated alpaca-zh sets are clean single-turn ZH instructions
# the base actually answers (esp. now that the harvest applies the chat template).
ZH_SOURCES = [
    ("llm-wizard/alpaca-gpt4-data-zh", "train", _alpaca_zh),
    ("shibing624/alpaca-zh", "train", _alpaca_zh),
]


def zh_classify(text):
    """Coarse ZH bucket, or None. Requires actual CJK content (so EN rows in a mixed
    dump don't leak in). Keyword-light — a subset only needs language coverage."""
    t = text.strip()
    if len(t) < 6 or len(text) > 2000 or not _CJK.search(text):
        return None
    if "```" in text or "def " in text or any(
        k in t for k in ("写一个函数", "写一个程序", "写一段代码", "编写程序", "编写一个函数",
                         "实现一个函数", "用python", "用 python", "编写代码")
    ):
        return "zh-code"
    if re.search(r"\d+\s*[+\-*/]\s*\d+", t) or any(
        k in t for k in ("计算", "求解", "求出", "解方程", "的概率", "的面积", "的体积")
    ):
        return "zh-math"
    if any(k in t for k in ("如何", "为什么", "什么是", "怎样", "怎么", "解释",
                            "比较", "区别", "介绍一下", "是什么")):
        return "zh-qa"
    return "zh-chat"

# ---- heuristic classifier ----------------------------------------------------
_CODE = re.compile(r"```|\bdef \b|\bclass \w+\(|\bSELECT\b|\bregex\b|#include\b")
_MATH = re.compile(
    r"\b(solve|calculate|compute|evaluate|simplify|integral|derivative|probability|"
    r"factorial|equation|\bprove that\b)\b|\d+\s*[+\-*/^]\s*\d+|\bwhat is \d"
)


def classify(text):
    """Map an instruction to a taxonomy category, or None to skip. Order = most
    specific first; the eval cats deliberately sit below the sharper task verbs so
    'summarize this essay' -> summarization, not chat-essay."""
    t = text.lower().strip()
    if len(t) < 16 or len(text) > 2000:
        return None
    if _CODE.search(text) or any(
        k in t for k in ("write a function", "write a program", "implement a",
                         "write code", "python function", "in javascript", "sql query",
                         "debug", "stack trace", "algorithm to", "time complexity")
    ):
        return "code-generation"
    if _MATH.search(t):
        return "math-reasoning"
    if t.startswith(("translate", "how do you say")) or "translate the" in t or (
        "in french" in t or "in spanish" in t or "into english" in t
    ):
        return "translation"
    if any(k in t for k in ("summarize", "summarise", "tl;dr", "give me a summary",
                            "in one sentence", "main points of")):
        return "summarization"
    if any(k in t for k in ("rewrite", "rephrase", "paraphrase", "fix the grammar",
                            "correct the grammar", "make this more", "improve the wording",
                            "proofread", "edit the following")):
        return "rewriting"
    if any(k in t for k in ("extract", "pull out", "identify all", "list all the",
                            "find all the", "parse the following")):
        return "extraction"
    if any(k in t for k in ("classify", "categorize", "categorise", "what sentiment",
                            "positive or negative", "label the", "is this spam")):
        return "classification"
    if any(k in t for k in ("write a story", "write a short story", "write a poem",
                            "continue the story", "continue the following", "write a tale",
                            "creative writing", "write a fictional", "opening of a story",
                            "write a narrative")):
        return "prose-continuation"
    if t.startswith(("pretend you", "act as", "you are a ", "roleplay", "imagine you are")) or (
        "in the style of" in t or "respond as" in t
    ):
        return "roleplay-dialogue"
    if any(k in t for k in ("brainstorm", "give me ideas", "some ideas for",
                            "suggest some", "name some", "list some ideas")):
        return "brainstorming"
    if t.startswith(("how do i", "how to", "how can i", "what's the best way to",
                     "steps to", "guide to")) or "step-by-step guide" in t:
        return "planning-howto"
    if any(k in t for k in ("write an essay", "essay about", "write a blog",
                            "write a detailed", "discuss the", "write about ")):
        return "chat-essay"
    if t.startswith(("explain", "how does", "how do ", "why does", "why is", "why are",
                     "what is", "what are", "compare ", "describe how", "analyze",
                     "what causes", "what's the difference")):
        return "analytical-qa"
    if t.startswith(("who ", "when ", "where ", "which ", "name the", "what year",
                     "what's the capital")) and len(t) < 160:
        return "factual-qa"
    if any(k in t for k in ("list ", "enumerate", "give me a list")):
        return "structured-list"  # natural list reqs (templates dominate this cat)
    return "open-chat"


def stream_hf(targets, templ_reserve, max_scan_per_source=400000):
    """Stream SOURCES, classify first-user turns into per-cat buckets up to cap."""
    from datasets import load_dataset

    # aperture/eval-HF cats fill fully from HF (unlimited supply); the template-fill
    # cats reserve half their target for templates, half pulled from natural prompts.
    caps = {c: targets[c] for c in targets}
    for c in TEMPLATE_FILL:
        caps[c] = int(targets[c] * templ_reserve)
    buckets = {c: [] for c in caps}
    seen = set()
    for repo, split, adapter in SOURCES:
        if all(len(buckets[c]) >= caps[c] for c in caps):
            break
        print(f"  streaming {repo} [{split}] ...", file=sys.stderr)
        try:
            ds = load_dataset(repo, split=split, streaming=True)
        except Exception as e:  # noqa: BLE001
            print(f"    skip {repo}: {e}", file=sys.stderr)
            continue
        scanned = got = 0
        for row in ds:
            if scanned >= max_scan_per_source:
                break
            scanned += 1
            prompt = adapter(row)
            if not prompt:
                continue
            cls = classify(prompt)
            if cls not in caps or len(buckets[cls]) >= caps[cls]:
                continue
            key = prompt[:200]
            if key in seen:
                continue
            seen.add(key)
            buckets[cls].append(prompt)
            got += 1
            if all(len(buckets[c]) >= caps[c] for c in caps):
                break
        print(f"    {repo}: scanned {scanned}, kept {got}", file=sys.stderr)
    return buckets


def stream_zh(zh_targets, max_scan_per_source=600000):
    """Stream ZH_SOURCES, classify CJK first-turns into coarse zh buckets up to cap."""
    from datasets import load_dataset

    buckets = {c: [] for c in zh_targets}
    seen = set()
    for repo, split, adapter in ZH_SOURCES:
        if all(len(buckets[c]) >= zh_targets[c] for c in zh_targets):
            break
        print(f"  streaming {repo} [{split}] (zh) ...", file=sys.stderr)
        try:
            ds = load_dataset(repo, split=split, streaming=True)
        except Exception as e:  # noqa: BLE001
            print(f"    skip {repo}: {e}", file=sys.stderr)
            continue
        scanned = got = 0
        for row in ds:
            if scanned >= max_scan_per_source:
                break
            scanned += 1
            prompt = adapter(row)
            if not prompt:
                continue
            cls = zh_classify(prompt)
            if cls not in zh_targets or len(buckets[cls]) >= zh_targets[cls]:
                continue
            key = prompt[:200]
            if key in seen:
                continue
            seen.add(key)
            buckets[cls].append(prompt)
            got += 1
            if all(len(buckets[c]) >= zh_targets[c] for c in zh_targets):
                break
        print(f"    {repo}: scanned {scanned}, kept {got}", file=sys.stderr)
    return buckets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=12000)
    ap.add_argument(
        "--zh-frac", type=float, default=0.0,
        help="fraction of the set that is Chinese (coarse zh-* buckets). EN cats are "
        "scaled to (1-zh_frac). 0 = English-only. ~0.15 keeps the head bilingual.",
    )
    ap.add_argument(
        "--templ-reserve", type=float, default=0.5,
        help="for the 2 template-fill cats (structured-list, prose): fraction pulled "
        "from natural HF prompts; the rest is templated. Aperture cats are 100% HF.",
    )
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dry-run", action="store_true", help="print distribution, no write")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    en_n = round(args.n * (1 - args.zh_frac))
    w_sum = sum(WEIGHTS.values())
    targets = {c: round(en_n * w / w_sum) for c, w in WEIGHTS.items()}
    zh_targets = {}
    if args.zh_frac > 0:
        zn, zw = args.n - en_n, sum(ZH_WEIGHTS.values())
        zh_targets = {c: round(zn * w / zw) for c, w in ZH_WEIGHTS.items()}

    hf = stream_hf(targets, args.templ_reserve)
    zh = stream_zh(zh_targets) if zh_targets else {}

    prompts, classes, seen = [], [], set()
    for cls in ALL_CATS:
        target = targets[cls]
        pool = []
        for p in hf.get(cls, []):
            if len(pool) >= target:
                break
            k = p[:200]
            if k not in seen:
                seen.add(k)
                pool.append(p)
        # templated fill: only the TEMPLATE_FILL cats; aperture cats accept HF underfill
        if cls in TEMPLATE_FILL and len(pool) < target:
            tries = 0
            while len(pool) < target and tries < target * 80:
                tries += 1
                p = render_class(cls, rng)
                if p not in seen:
                    seen.add(p)
                    pool.append(p)
        for p in pool:
            prompts.append(p)
            classes.append(cls)

    # ZH buckets (no template fill — accept HF underfill)
    for cls in ZH_CATS:
        for p in zh.get(cls, [])[: zh_targets.get(cls, 0)]:
            k = p[:200]
            if k not in seen:
                seen.add(k)
                prompts.append(p)
                classes.append(cls)

    order = list(range(len(prompts)))
    rng.shuffle(order)
    prompts = [prompts[i] for i in order]
    classes = [classes[i] for i in order]

    report_cats = ALL_CATS + (ZH_CATS if zh_targets else [])
    all_targets = {**targets, **zh_targets}
    counts = {c: classes.count(c) for c in report_cats}
    print(f"\n{'category':22s} {'count':>6s} {'target':>6s}  {'%':>5s}  src")
    for c in report_cats:
        src = "HF+templ" if c in TEMPLATE_FILL else ("ZH" if c in ZH_CATS else "HF")
        tag = "  [EVAL]" if c in EVAL_CATS else ""
        print(f"{c:22s} {counts[c]:6d} {all_targets.get(c, 0):6d}  "
              f"{counts[c] / max(1, len(prompts)) * 100:4.1f}%  {src}{tag}")
    zh_n = sum(counts[c] for c in ZH_CATS if c in counts)
    print(f"\ntotal: {len(prompts)} prompts ({len(prompts) - zh_n} EN / {zh_n} ZH = "
          f"{zh_n / max(1, len(prompts)) * 100:.0f}% zh) across "
          f"{sum(1 for c in counts.values() if c)} cats")

    if args.dry_run:
        print("(dry-run: nothing written)")
        return 0

    with open(args.out, "wb") as f:
        f.write(b"\x00".join(p.encode() for p in prompts) + b"\x00")
    with open(args.out + ".classes.json", "w") as f:
        json.dump({"classes": classes}, f)
    print(f"\nwrote {len(prompts)} prompts -> {args.out} (+ .classes.json)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
