import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path
 
from dotenv import load_dotenv
 
try:
    from groq import APIConnectionError, APIStatusError, Groq, RateLimitError
except ImportError:
    sys.exit("Missing dependency: run `pip install -r requirements.txt`")
 
try:
    from hindsight_client import Hindsight
except ImportError:
    Hindsight = None  # falls back to the local simulator
 
load_dotenv()
 
PRIMARY_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
FALLBACK_MODEL = "qwen/qwen3-32b"
DATA_PATH = Path(__file__).parent / "data" / "northwind_deal.json"
 
QUERY = (
    "I have a final pricing call with Northwind Freight tomorrow. Brief me: "
    "key risks, how to approach each stakeholder, what I still owe them, "
    "and the single best next action."
)
SYSTEM_PROMPT = (
    "You are DealPilot, a senior sales-intelligence assistant. Give concise, "
    "actionable pre-call briefs (max ~180 words). Use only facts you are given; "
    "if context is missing, say so instead of inventing details."
)
DEFAULT_FEEDBACK = [
    "Rep feedback: Meera Kapoor (VP Ops, champion) never reads long decks; "
    "keep everything to 2 slides or a short message.",
    "Rep feedback: Arjun Rao (CFO) ignored a generic pricing deck but replied to a "
    "one-page ROI sheet built from Northwind's own fuel spend. Do not lead with discounts.",
    "Rep feedback: I promised Dev Malhotra (IT Security) the data residency answer "
    "and missed the deadline once already. This is an open commitment.",
]
 
 
# --------------------------------------------------------------------------- #
# Terminal styling
# --------------------------------------------------------------------------- #
class C:
    on = sys.stdout.isatty() and not os.getenv("NO_COLOR")
    RESET, BOLD = "\033[0m", "\033[1m"
    RED, GREEN, YELLOW = "\033[91m", "\033[92m", "\033[93m"
    BLUE, MAGENTA, CYAN, GREY = "\033[94m", "\033[95m", "\033[96m", "\033[90m"
 
    @classmethod
    def s(cls, text: str, *styles: str) -> str:
        return f"{''.join(styles)}{text}{cls.RESET}" if cls.on else text
 
 
def banner(title: str, color: str = C.CYAN) -> None:
    line = "=" * 72
    print(f"\n{C.s(line, color)}\n{C.s(title, C.BOLD, color)}\n{C.s(line, color)}")
 
 
def info(msg: str) -> None:
    print(C.s(f"  {msg}", C.GREY))
 
 
def warn(msg: str) -> None:
    print(C.s(f"  [warn] {msg}", C.YELLOW))
 
 
# --------------------------------------------------------------------------- #
# Memory layer: Hindsight with a local simulator as a safety net
# --------------------------------------------------------------------------- #
STOPWORDS = {
    "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "with", "is",
    "are", "was", "i", "me", "my", "have", "has", "what", "how", "each", "still",
}
 
 
def _tokens(text: str) -> set:
    return {w for w in re.findall(r"[a-z0-9$]+", text.lower()) if w not in STOPWORDS}
 
 
class LocalMemory:
    """Minimal in-process stand-in for Hindsight (keyword-overlap retrieval)."""
 
    def __init__(self) -> None:
        self.items: list[str] = []
 
    def retain(self, text: str) -> None:
        self.items.append(text)
 
    def recall(self, query: str, top_k: int = 8) -> list[str]:
        q = _tokens(query)
        scored = [(len(q & _tokens(t)), t) for t in self.items]
        scored = [s for s in scored if s[0] > 0]
        scored.sort(key=lambda s: s[0], reverse=True)
        return [t for _, t in scored[:top_k]]
 
 
class Memory:
    """
    Writes to Hindsight and mirrors locally. Reads from Hindsight and falls back
    to the local mirror if the service is unreachable or returns nothing yet
    (Hindsight indexes asynchronously, so we retry briefly first).
    """
 
    def __init__(self, bank_id: str, force_local: bool = False) -> None:
        self.bank_id = bank_id
        self.local = LocalMemory()
        self.client = None
        self.source = "local-simulator"
        api_key = os.getenv("HINDSIGHT_API_KEY")
        if force_local or not api_key or Hindsight is None:
            reason = "--local flag" if force_local else "no HINDSIGHT_API_KEY / SDK not installed"
            warn(f"Using local memory simulator ({reason}).")
            return
        try:
            self.client = Hindsight(
                base_url=os.getenv("HINDSIGHT_BASE_URL", "https://api.hindsight.vectorize.io"),
                api_key=api_key,
            )
            self.source = "hindsight"
        except Exception as err:
            warn(f"Could not initialise Hindsight ({err}). Using local simulator.")
 
    def retain(self, text: str, context: str = "deal note") -> None:
        self.local.retain(text)
        if not self.client:
            return
        try:
            self.client.retain(bank_id=self.bank_id, content=text, context=context)
        except Exception as err:
            warn(f"Hindsight retain failed ({type(err).__name__}: {err}); kept in local mirror.")
 
    def recall(self, query: str, attempts: int = 3, wait: float = 3.0) -> tuple[list[str], str]:
        """Returns (memories, source_label)."""
        if self.client:
            for i in range(attempts):
                try:
                    result = self.client.recall(bank_id=self.bank_id, query=query)
                    items = getattr(result, "results", result) or []
                    texts = [str(getattr(x, "text", x)) for x in items]
                    if texts:
                        return texts, "Hindsight"
                except Exception as err:
                    warn(f"Hindsight recall failed ({type(err).__name__}: {err}).")
                    break
                if i < attempts - 1:
                    info(f"Waiting for Hindsight indexing ({i + 1}/{attempts - 1})...")
                    time.sleep(wait)
            if self.local.items:
                warn("Falling back to local memory mirror for recall.")
        return self.local.recall(query), "local mirror"
 
 
# --------------------------------------------------------------------------- #
# LLM layer: Groq with retries, backoff and model fallback
# --------------------------------------------------------------------------- #
class LLM:
    def __init__(self) -> None:
        key = os.getenv("GROQ_API_KEY")
        if not key:
            sys.exit("GROQ_API_KEY is not set. Copy .env.example to .env and add your key.")
        self.client = Groq(api_key=key)
 
    def ask(self, user_prompt: str, retries: int = 4) -> str:
        models = [PRIMARY_MODEL, FALLBACK_MODEL]
        last_err: Exception | None = None
        for attempt in range(retries):
            model = models[0] if attempt < 2 else models[1]
            try:
                resp = self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.2,
                    max_tokens=900,
                )
                text = (resp.choices[0].message.content or "").strip()
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
                if not text:
                    raise ValueError("empty completion")
                return text
            except RateLimitError as err:
                last_err, delay = err, 2 ** (attempt + 1)
                warn(f"Rate limited on {model}; retrying in {delay}s.")
            except APIConnectionError as err:
                last_err, delay = err, 2 ** attempt
                warn(f"Network error talking to Groq; retrying in {delay}s.")
            except APIStatusError as err:
                last_err, delay = err, 1
                if err.status_code in (401, 403):
                    sys.exit("Groq rejected your API key (401/403). Check GROQ_API_KEY.")
                warn(f"Groq API error {err.status_code} on {model}; trying next attempt.")
            except Exception as err:  # malformed/empty responses etc.
                last_err, delay = err, 1
                warn(f"Unexpected LLM error on {model}: {err}")
            time.sleep(delay)
        raise RuntimeError(f"LLM failed after {retries} attempts: {last_err}")
 
 
# --------------------------------------------------------------------------- #
# Agent
# --------------------------------------------------------------------------- #
def load_deal_history() -> list[str]:
    """Turn the synthetic dataset into memory-ready sentences."""
    if not DATA_PATH.exists():
        warn(f"{DATA_PATH} not found; skipping deal history.")
        return []
    try:
        data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
        p = data["prospect"]
        notes = [
            f"Deal profile: {p['company']} ({p['industry']}), fleet {p['fleet_size']}, "
            f"${p['deal_value_usd']} ARR, stage {p['stage']}. Rep: {data['rep']}. "
            + "; ".join(f"{s['name']} ({s['role']}, {s['stance']})" for s in data["stakeholders"])
        ]
        notes += [
            f"Past deal {d['company']} was {d['outcome']}. Lesson: {d['lesson']}"
            for d in data["past_deals"]
        ]
        notes += [
            f"[{i['date']}] {i['type']} with {p['company']}: {i['notes']}"
            for i in data["interactions"]
        ]
        return notes
    except (KeyError, json.JSONDecodeError) as err:
        warn(f"Could not parse deal data ({err}); skipping deal history.")
        return []
 
 
def show_recall(memories: list[str], source: str) -> None:
    print(C.s(f"\n[HINDSIGHT MEMORY RECALL] via {source}: {len(memories)} memories", C.BOLD, C.MAGENTA))
    if not memories:
        print(C.s("  (nothing recalled: the agent has no prior context)", C.MAGENTA))
    for m in memories:
        print(C.s(f"  * {m[:150]}{'...' if len(m) > 150 else ''}", C.MAGENTA))
 
 
def main() -> None:
    ap = argparse.ArgumentParser(description="DealPilot: deal intelligence with Hindsight memory")
    ap.add_argument("--interactive", action="store_true", help="type your own feedback in step 2")
    ap.add_argument("--local", action="store_true", help="use the local memory simulator")
    ap.add_argument("--bank", default=None, help="reuse a specific Hindsight memory bank id")
    args = ap.parse_args()
 
    bank_id = args.bank or f"dealpilot-demo-{uuid.uuid4().hex[:8]}"
    llm = LLM()
    memory = Memory(bank_id, force_local=args.local)
    info(f"Memory backend: {memory.source} | bank: {bank_id} | model: {PRIMARY_MODEL}")
    print(C.s(f"\nQuery used in every interaction:\n  \"{QUERY}\"", C.BOLD))
 
    # ---- Interaction 1: cold query -------------------------------------- #
    banner("INTERACTION 1: COLD QUERY (no memory yet)", C.RED)
    memories, source = memory.recall(QUERY, attempts=1)
    show_recall(memories, source)
    try:
        answer = llm.ask(QUERY if not memories else f"Context:\n" + "\n".join(memories) + f"\n\n{QUERY}")
    except RuntimeError as err:
        sys.exit(C.s(f"Fatal: {err}", C.RED, C.BOLD))
    print(C.s("\nAgent response (generic):", C.BOLD, C.RED))
    print(answer)
 
    # ---- Interaction 2: feedback -> memory ------------------------------ #
    banner("INTERACTION 2: FEEDBACK & HISTORY STORED IN MEMORY", C.YELLOW)
    feedback = list(DEFAULT_FEEDBACK)
    if args.interactive:
        typed = input(C.s("Enter a preference/feedback note (blank = use defaults): ", C.BOLD)).strip()
        if typed:
            feedback = [f"Rep feedback: {typed}"]
    to_store = load_deal_history() + feedback
    for n, note in enumerate(to_store, 1):
        memory.retain(note, context="rep feedback" if note.startswith("Rep feedback") else "deal history")
        print(C.s(f"  [RETAINED {n}/{len(to_store)}]", C.BOLD, C.GREEN), note[:110] + ("..." if len(note) > 110 else ""))
    info("Memory retained. Hindsight processes memories asynchronously.")
    time.sleep(3 if memory.client else 0)
 
    # ---- Interaction 3: same query, memory-powered ---------------------- #
    banner("INTERACTION 3: SAME QUERY, NOW WITH MEMORY", C.GREEN)
    memories, source = memory.recall(QUERY)
    show_recall(memories, source)
    context = "\n".join(f"- {m}" for m in memories) or "(no memories available)"
    try:
        answer = llm.ask(f"Relevant memory:\n{context}\n\nTask: {QUERY}")
    except RuntimeError as err:
        sys.exit(C.s(f"Fatal: {err}", C.RED, C.BOLD))
    print(C.s("\nAgent response (memory-tailored):", C.BOLD, C.GREEN))
    print(answer)
    print(C.s("\nDone. Compare Interaction 1 vs 3: same question, different quality.\n", C.BOLD))
 
 
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
 
