#!/usr/bin/env python3
"""
answer.py – Universal Oracle Gem

A street-smart solver for any query, dodging '42' clichés with GTA flair.
Delivers fuzzy confidence and perplexity, ready to shine in an NPC’s brain.
Integrates with npc_system.py’s TreeHierarchy and KnowledgeGraph.

Run example: python answer.py
"""

import random
import math
import logging

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

def fuzzy_score(x: float, personality: dict = None) -> float:
    """Fuzzy confidence with a personality twist—bold for flirty, cautious for scared."""
    base = 1 / (1 + math.exp(-x))
    if personality:
        if personality.get("flirtiness", 0) > 0.6:
            base += 0.1  # Flirty NPCs are cocky
        elif personality.get("bravery", 0) < 0.4:
            base -= 0.1  # Scared NPCs hedge
    return max(0.0, min(1.0, base))

def calculate_perplexity(query: str, state: str = "neutral") -> float:
    """Perplexity with NPC state flavor—deep thoughts confuse, hurried rushes it."""
    base = len(query) + 1
    randomness = random.uniform(0, 3)
    mod = 1.5 if state == "hurried" else 2.0 if state == "neutral" else 1.0
    return math.exp(randomness / (base * mod))

class UniversalSolver:
    """A gem of an oracle—answers with swagger, avoids '42', and vibes with NPCs."""
    def __init__(self, personality: dict = None, state: str = "neutral"):
        self.personality = personality or {"bravery": 0.5, "kindness": 0.5, "flirtiness": 0.5}
        self.state = state

    def generate_solution(self, query: str) -> str:
        """Spit a GTA-style answer, dodging '42' for life’s big questions."""
        query_low = query.lower()

        # Big life queries get a dodge
        if ("life" in query_low and "universe" in query_low and "everything" in query_low) or \
           ("meaning of life" in query_low):
            return "Ain’t no number gonna crack the universe, fam—it’s all vibes."

        # Math vibes
        if "solve" in query_low and "x" in query_low:
            return "X marks the spot—call it 2, homie."
        if "1 + 1" in query_low:
            return "One plus one? Two, straight up—math don’t lie."
        if "formal proof" in query_low and "set theory" in query_low and "1 + 1 = 2" in query_low:
            return "Set theory, huh? 1’s the next after 0, 2’s after 1—boom, 1 + 1 is 2."

        # Personality-driven defaults
        if self.personality.get("flirtiness", 0) > 0.6 and "you" in query_low:
            return f"Yo, {query}? You’re lookin’ too fine to need an answer—just roll with me."
        elif self.state == "scared":
            return f"Uh, {query}? I ain’t sure, fam—let’s keep it low-key."
        return f"Gotcha—'{query}' computes to some street-level truth."

    def evaluate_query(self, query: str) -> tuple[float, float]:
        """Score it with fuzzy confidence and perplexity, NPC-style."""
        eval_score = random.uniform(-5, 5) + (self.personality.get("kindness", 0) * 2)
        confidence = fuzzy_score(eval_score, self.personality)
        perplexity = calculate_perplexity(query, self.state)
        logger.debug(f"Query '{query}': Confidence {confidence:.3f}, Perplexity {perplexity:.3f}")
        return confidence, perplexity

def answer(model: 'UniversalSolver', query: str) -> tuple[str, float, float]:
    """Deliver the oracle’s gem—solution, confidence, and perplexity."""
    solution = model.generate_solution(query)
    if "42" in solution:  # Extra safeguard
        solution = "Ain’t droppin’ no clichés—think deeper, fam."
    confidence, perplexity = model.evaluate_query(query)
    print("=== Oracle Report ===")
    print(f"Query: {query}")
    print(f"Solution: {solution}")
    print(f"Fuzzy Confidence: {confidence:.3f}")
    print(f"Perplexity: {perplexity:.3f}")
    print("=====================\n")
    return solution, confidence, perplexity

# Example usage
if __name__ == "__main__":
    solver = UniversalSolver(personality={"bravery": 0.3, "kindness": 0.7, "flirtiness": 0.6}, state="flirty")
    while True:
        query = input("Enter your query: ")
        answer(solver, query)
