# -*- coding: utf-8 -*-
"""
Halting Solver: Predicts if Python code halts using static analysis and probabilistic heuristics.
License: Apache-2.0
"""

import ast
import random
from typing import Dict, Tuple, Optional
from collections import defaultdict
import numpy as np
from pydantic import BaseModel, Field

class HaltingConfig(BaseModel):
    """Configuration for HaltingSolver."""
    heuristic_threshold: float = Field(0.5, ge=0.0, le=1.0)
    max_recursion_depth: int = Field(10, ge=1)
    seed: Optional[int] = None

class HaltingResult(BaseModel):
    """Result of halting analysis with explanation."""
    prediction: str  # "Halts", "Doesn't Halt", "Uncertain"
    confidence: float  # 0 to 1
    explanation: Dict  # Features and weights

class HaltingSolver:
    """Analyzes Python code to predict halting behavior."""
    def __init__(self, config: Optional[HaltingConfig] = None):
        self.config = config or HaltingConfig()
        if self.config.seed is not None:
            random.seed(self.config.seed)
            np.random.seed(self.config.seed)

    def extract_features(self, code: str) -> Dict[str, float]:
        """
        Extract syntactic and semantic features from code using AST.
        Returns normalized feature weights contributing to non-halting risk.
        """
        features = defaultdict(float)
        try:
            tree = ast.parse(code)
            # Detect functions and their call graph
            functions = {}
            calls = defaultdict(list)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions[node.name] = node
                    features["functions"] += 1
                elif isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name):
                        calls[node.func.id].append(node)
                    features["calls"] += 1

            # Analyze loops
            for node in ast.walk(tree):
                if isinstance(node, ast.While):
                    features["loops"] += 1
                    if isinstance(node.test, ast.Constant) and node.test.value is True:
                        features["infinite_loops"] += 1
                elif isinstance(node, ast.For):
                    features["loops"] += 1
                elif isinstance(node, ast.Return):
                    features["returns"] += 1

            # Check for recursion
            for func_name, node in functions.items():
                for call in calls.get(func_name, []):
                    # Simplified: Assume calls within function body indicate recursion
                    if any(n for n in ast.walk(node) if n == call):
                        features["recursion"] += 1

            # Normalize features (weights reflect non-halting risk)
            total = sum(features.values()) or 1
            return {k: min(v / total, 1.0) for k, v in features.items()}
        except SyntaxError:
            return {"syntax_error": 1.0}

    def probabilistic_simulation(self, features: Dict[str, float]) -> Tuple[str, float]:
        """
        Simulate probabilistic evaluation based on features.
        Returns (prediction, confidence).
        """
        # Base probability of non-halting increases with risky features
        base_prob = features.get("infinite_loops", 0) * 0.5 + features.get("recursion", 0) * 0.3
        prob_non_halting = min(base_prob, 0.95)  # Cap to avoid overconfidence
        prob_halting = 1 - prob_non_halting

        # Simulate with randomness for uncertainty
        choice = np.random.choice(
            ["Doesn't Halt", "Halts"],
            p=[prob_non_halting, prob_halting]
        )
        confidence = prob_non_halting if choice == "Doesn't Halt" else prob_halting
        return choice, max(confidence, 0.1)  # Minimum confidence for realism

    def heuristic_score(self, features: Dict[str, float]) -> float:
        """
        Compute heuristic score (higher = more likely to not halt).
        """
        score = (
            features.get("infinite_loops", 0) * 0.4 +
            features.get("recursion", 0) * 0.3 +
            features.get("loops", 0) * 0.2 +
            features.get("calls", 0) * 0.1
        )
        return min(score, 1.0)

    def analyze(self, code: str) -> HaltingResult:
        """
        Predict halting behavior with confidence and explanation.
        Returns HaltingResult with prediction, confidence, and feature weights.
        """
        # Step 1: Extract features
        features = self.extract_features(code)
        if features.get("syntax_error"):
            return HaltingResult(
                prediction="Uncertain",
                confidence=0.1,
                explanation={"syntax_error": "Invalid Python code"}
            )

        # Step 2: Probabilistic simulation
        sim_prediction, sim_confidence = self.probabilistic_simulation(features)

        # Step 3: Heuristic score
        heuristic_score = self.heuristic_score(features)

        # Step 4: Decision logic
        if features.get("infinite_loops", 0) > 0 and heuristic_score > self.config.heuristic_threshold:
            prediction = "Doesn't Halt"
            confidence = min(0.9 + sim_confidence * 0.1, 0.95)
            explanation = {
                "reason": "Infinite loop detected",
                "heuristic_score": heuristic_score,
                "features": dict(features)
            }
        elif features.get("loops", 0) == 0 and features.get("recursion", 0) == 0:
            prediction = "Halts"
            confidence = min(0.8 + sim_confidence * 0.2, 0.9)
            explanation = {
                "reason": "No loops or recursion detected",
                "heuristic_score": heuristic_score,
                "features": dict(features)
            }
        else:
            prediction = sim_prediction if sim_confidence > 0.5 else "Uncertain"
            confidence = sim_confidence if prediction != "Uncertain" else 0.5
            explanation = {
                "reason": "Complex control flow; probabilistic decision",
                "heuristic_score": heuristic_score,
                "features": dict(features)
            }

        return HaltingResult(
            prediction=prediction,
            confidence=confidence,
            explanation=explanation
        )

# Demo
def demonstrate_halting_solver():
    """Demonstrate the halting solver on sample codes."""
    solver = HaltingSolver(HaltingConfig(seed=42))
    samples = [
        """
def example():
    while True:
        pass
""",
        """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)
""",
        """
x = 0
for i in range(10):
    x += i
"""
    ]

    for code in samples:
        result = solver.analyze(code)
        print("\nCode:")
        print(code.strip())
        print(f"Prediction: {result.prediction}")
        print(f"Confidence: {result.confidence:.2f}")
        print("Explanation:")
        for key, value in result.explanation.items():
            print(f"  {key}: {value}")

if __name__ == "__main__":
    demonstrate_halting_solver()
