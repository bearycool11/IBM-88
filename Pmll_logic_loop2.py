# -*- coding: utf-8 -*-
"""
PMLL: Persistent Memory Learning Loop
Tracks novel inputs, logs state, and persists progress with semantic awareness.
License: Apache-2.0
"""

import json
import logging
import os
from typing import List, Optional
from difflib import SequenceMatcher
from pydantic import BaseModel, Field
import sys

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("pmll_log.txt"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("PMLL")

class State(BaseModel):
    """Structured state for PMLL with validation."""
    iteration_count: int = 0
    value: float = 1.0
    input_history: List[str] = []
    novelty_threshold: float = Field(0.7, ge=0.0, le=1.0)
    growth_rate: float = Field(0.1, ge=0.0)

class PMLL:
    """Manages a persistent learning loop with novelty detection."""
    def __init__(self, state_file: str = "pmll_state.json"):
        self.state_file = state_file
        self.state = State()
        self.load_state()

    def load_state(self):
        """Load state from JSON file if it exists."""
        try:
            if os.path.exists(self.state_file):
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.state = State(**data)
                logger.info(
                    f"Loaded state: Iteration {self.state.iteration_count}, "
                    f"Value {self.state.value:.2f}, "
                    f"History size {len(self.state.input_history)}"
                )
            else:
                logger.info("Initialized new state")
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to load state: {e}. Starting fresh.")
            self.state = State()

    def save_state(self):
        """Save state to JSON file."""
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state.dict(), f, indent=2)
            logger.debug("State saved")
        except IOError as e:
            logger.error(f"Failed to save state: {e}")

    def is_novel(self, input_str: str) -> bool:
        """
        Check if input is novel based on semantic similarity to history.
        Returns True if novel, False otherwise.
        """
        if not input_str or not self.state.input_history:
            return True
        # Compare with recent inputs (last 5 for efficiency)
        for prev_input in self.state.input_history[-5:]:
            similarity = SequenceMatcher(None, input_str.lower(), prev_input.lower()).ratio()
            if similarity > self.state.novelty_threshold:
                logger.debug(f"Input '{input_str}' similar to '{prev_input}' (score: {similarity:.2f})")
                return False
        return True

    def process_input(self, input_str: str):
        """Process a single input, update state, and log."""
        self.state.iteration_count += 1
        is_novel = self.is_novel(input_str)

        # Update value based on novelty
        if is_novel:
            self.state.value *= (1 + self.state.growth_rate * 1.5)  # Bonus for novelty
            self.state.input_history.append(input_str)
            logger.info(f"Novel input detected: '{input_str}'")
        else:
            self.state.value *= (1 + self.state.growth_rate * 0.5)  # Smaller growth
            logger.info(f"Repeated input: '{input_str}'")

        # Cap history to prevent bloat
        if len(self.state.input_history) > 100:
            self.state.input_history = self.state.input_history[-50:]

        # Log state
        log_msg = (
            f"Iteration: {self.state.iteration_count}, "
            f"Value: {self.state.value:.2f}, "
            f"Input: '{input_str}'"
        )
        logger.info(log_msg)
        print(log_msg)

    def run(self):
        """Run the interactive input loop."""
        print("PMLL: Enter text (type 'exit' to quit, 'history' to view inputs)")
        while True:
            try:
                input_str = input("> ").strip()
                if input_str.lower() == "exit":
                    logger.info("Exiting loop")
                    self.save_state()
                    break
                elif input_str.lower() == "history":
                    if self.state.input_history:
                        print("Input History:")
                        for i, inp in enumerate(self.state.input_history[-10:], 1):
                            print(f"  {i}. {inp}")
                    else:
                        print("No history yet")
                    continue
                if not input_str:
                    logger.debug("Empty input ignored")
                    continue

                self.process_input(input_str)
                self.save_state()

            except (KeyboardInterrupt, EOFError):
                logger.info("Interrupted by user")
                self.save_state()
                break
            except Exception as e:
                logger.error(f"Error processing input: {e}")
                print(f"Oops, something broke: {e}. Try again!")

if __name__ == "__main__":
    pmll = PMLL()
    pmll.run()
