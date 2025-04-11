#!/usr/bin/env python3
"""
npc_system.py – Rockstar Hybrid v2.2

An AI-driven NPC system for a GTA-like open-world game, built from scratch.
No external deps—just pure Python with dynamic world, persistent memory, and snappy dialogue.
Combines the best of v2.0 (world, persistence) and v2.1 (homebrew, lazy-friendly).

Run game loop: python npc_system.py
Run tests: python npc_system.py test
"""

import json
import os
import random
import sys
import time
import unittest
from collections import deque
from dataclasses import dataclass, field
from textwrap import dedent
import logging

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("npc_system.log")]
)
logger = logging.getLogger(__name__)

# Models
@dataclass
class MemoryEntry:
    """A single memory entry."""
    timestamp: float
    player_text: str
    npc_reply: str
    sentiment: float  # -1 (negative) to 1 (positive)

    def to_dict(self):
        return {"timestamp": self.timestamp, "player_text": self.player_text,
                "npc_reply": self.npc_reply, "sentiment": self.sentiment}

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

@dataclass
class NPC:
    """Dynamic NPC with personality, memory, and state."""
    npc_id: str
    personality: dict[str, float]  # bravery, kindness, flirtiness: 0-1
    routine: dict[int, str]  # hour -> task
    state: str = "neutral"
    location: str = "town_square"
    memory: deque = field(default_factory=lambda: deque(maxlen=5))
    reputation: float = 0.0  # -1 (hated) to 1 (loved)
    consent_level: str = "safe"  # safe, flirty, explicit
    memory_file: str = "npc_memory.json"

    def save_memory(self):
        """Persist memory to disk (optional)."""
        try:
            with open(self.memory_file, "w") as f:
                json.dump([m.to_dict() for m in self.memory], f, indent=2)
            logger.debug(f"Saved {len(self.memory)} memories for {self.npc_id}")
        except Exception as e:
            logger.warning(f"Failed to save memory: {e}")

    def load_memory(self):
        """Load memory from disk if exists."""
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, "r") as f:
                    data = json.load(f)
                    self.memory = deque([MemoryEntry.from_dict(d) for d in data],
                                      maxlen=self.memory.maxlen)
                logger.info(f"Loaded {len(self.memory)} memories for {self.npc_id}")
        except Exception as e:
            logger.warning(f"Failed to load memory: {e}")

class WorldState:
    """Dynamic world with simplified time and events."""
    def __init__(self):
        self.hour = 14
        self.weather = "sunny"
        self.time_of_day = "afternoon"
        self.tick_interval = 30  # seconds per hour
        self.events = [
            "A fight broke out nearby!",
            "Cops are chasing someone!",
            "Heard a deal at the docks...",
            "Music’s blasting down the block."
        ]

    def update(self) -> str:
        """Advance time and maybe trigger an event."""
        self.hour = (self.hour + 1) % 24
        self.time_of_day = (
            "morning" if 6 <= self.hour < 12 else
            "afternoon" if 12 <= self.hour < 18 else
            "evening" if 18 <= self.hour < 22 else
            "night"
        )
        weather_probs = {"sunny": 0.6, "rainy": 0.3, "stormy": 0.1}
        self.weather = random.choices(list(weather_probs.keys()), weights=weather_probs.values())[0]
        event = random.choice(self.events) if random.random() < 0.2 else ""
        logger.debug(f"World: {self.hour}:00, {self.time_of_day}, {self.weather}, {event or 'calm'}")
        return event

# Dialogue engine
def estimate_sentiment(text: str) -> float:
    """Guess sentiment from text (-1 to 1)."""
    text = text.lower()
    positive = ("help", "love", "cool", "nice", "thanks")
    negative = ("kill", "rob", "hate", "damn", "loser")
    score = 0.0
    if any(k in text for k in positive):
        score += 0.5
    if any(k in text for k in negative):
        score -= 0.5
    return max(-1.0, min(1.0, score))

def build_context(npc: NPC, player_text: str, world: WorldState, event: str) -> str:
    """Build compact dialogue context."""
    memories = [f"P: {m.player_text} | Me: {m.npc_reply}" for m in npc.memory]
    return dedent(f"""
        NPC: {npc.npc_id}
        Mood: {npc.state}
        Location: {npc.location}
        Task: {npc.routine.get(world.hour, 'chilling')}
        Reputation: {npc.reputation:.2f}
        Consent: {npc.consent_level}
        Memories: {'; '.join(memories) or 'None'}
        Time: {world.hour}:00 ({world.time_of_day})
        Weather: {world.weather}
        Event: {event or 'Nothing much'}
        Player: "{player_text}"
    """).strip()

def homebrew_dialogue(npc: NPC, player_text: str, world: WorldState) -> tuple[str, str]:
    """Homebrewed dialogue with personality, memory, and world context."""
    text = player_text.lower()
    sentiment = estimate_sentiment(text)
    event = world.update() if random.random() < 0.5 else ""  # 50% tick chance
    new_state = npc.state
    reply = ""

    # Consent logic
    if "consent" in text:
        if npc.reputation > 0 and npc.personality.get("flirtiness", 0) > 0.4:
            npc.consent_level = "flirty"
            reply = "Aight, I’m down to flirt—whatchu got?"
            new_state = "flirty"
        elif npc.reputation > 0.5:
            npc.consent_level = "explicit"
            reply = "Yo, we can get wild if you’re real about it."
            new_state = "flirty"
        else:
            reply = "Nah, let’s keep it chill for now."
    elif "stop" in text or "chill" in text:
        npc.consent_level = "safe"
        reply = "Got it, keeping it cool."
        new_state = "neutral"

    # State transitions
    state_probs = {
        "neutral": 0.4 + npc.personality.get("kindness", 0) * 0.2,
        "scared": 0.3 if npc.reputation < 0 else 0.1,
        "flirty": 0.3 if npc.consent_level in ("flirty", "explicit") else 0.0,
        "angry": 0.2 if npc.reputation < -0.3 else 0.1,
        "hurried": 0.2 if world.weather == "stormy" else 0.1
    }
    if random.random() < 0.3:  # 30% shift chance
        total = sum(state_probs.values())
        if total > 0:
            new_state = random.choices(
                list(state_probs.keys()),
                weights=[v / total for v in state_probs.values()]
            )[0]

    # Response logic
    if not reply:
        if "kill" in text or "rob" in text:
            new_state = "scared"
            reply = random.choice(["Yo, back off—I ain’t playin’!", "You serious? I’m gone!"])
        elif "help" in text:
            reply = random.choice(["Thanks, that’s solid.", "Appreciate the hand, fam."])
        elif any(k in text for k in ("love", "hot", "date")) and npc.consent_level != "safe":
            new_state = "flirty"
            reply = random.choice([
                "Oh, you’re bringing heat, huh?",
                "Keep that up, we’ll see where it goes!"
            ])
        elif event:
            reply = f"{event} What’s your deal today?"
        elif world.weather == "rainy" and random.random() < 0.5:
            reply = "This rain’s trash—got plans?"
        else:
            replies = {
                "neutral": ["What’s good, homie?", "Just vibing here."],
                "scared": ["Keep it calm, aight?", "I’m jumpy today."],
                "flirty": ["You lookin’ sharp—wanna roll?", "What’s the vibe?"],
                "angry": ["Not feelin’ it, bruh.", "What’s your problem?"],
                "hurried": ["Gotta move—talk fast!", "Weather’s rushing me!"]
            }
            reply = random.choice(replies.get(new_state, replies["neutral"]))

    # Update NPC
    npc.memory.append(MemoryEntry(
        timestamp=time.time(),
        player_text=player_text,
        npc_reply=reply,
        sentiment=sentiment
    ))
    npc.reputation += sentiment * 0.1
    npc.reputation = max(-1.0, min(1.0, npc.reputation))
    npc.save_memory()

    return reply, new_state

def fake_tts(text: str):
    """Simulate TTS with console flair."""
    print(f"🎙️ {text}")

def fake_speech_recognition() -> str:
    """Simulate speech with keyboard input."""
    try:
        text = input("🎤 You: ").strip()
        if random.random() < 0.1:
            text = text.replace("you", "ya") or text + " uh"
        return text
    except KeyboardInterrupt:
        return "exit"

# Main loop
def dialogue_loop(npc: NPC):
    """Run the NPC dialogue loop."""
    logger.info(f"Starting chat with {npc.npc_id}")
    print(f"Chat with {npc.npc_id} (type 'exit' to quit)")
    npc.load_memory()
    world = WorldState()

    while True:
        player_text = fake_speech_recognition()
        if player_text.lower() == "exit":
            logger.info("Player exited")
            npc.save_memory()
            break

        reply, new_state = homebrew_dialogue(npc, player_text, world)
        npc.state = new_state
        print(f"\n{npc.npc_id} [{npc.state}, rep: {npc.reputation:.2f}]:")
        fake_tts(reply)

# Tests
class TestNPCSystem(unittest.TestCase):
    def setUp(self):
        self.npc = NPC(
            npc_id="npc_001",
            personality={"bravery": 0.3, "kindness": 0.7, "flirtiness": 0.6},
            routine={8: "work", 12: "lunch", 18: "home"}
        )
        self.world = WorldState()

    def test_dialogue_basic(self):
        reply, state = homebrew_dialogue(self.npc, "Hey, what’s up?", self.world)
        self.assertIn(state, ("neutral", "scared", "flirty", "angry", "hurried"))
        self.assertTrue(reply)

    def test_sentiment(self):
        self.assertGreaterEqual(estimate_sentiment("I’ll help you"), 0)
        self.assertLessEqual(estimate_sentiment("I’ll rob you"), 0)

    def test_consent(self):
        self.npc.reputation = 0.5
        homebrew_dialogue(self.npc, "Consent to flirt", self.world)
        self.assertIn(self.npc.consent_level, ("flirty", "explicit"))
        homebrew_dialogue(self.npc, "Stop it", self.world)
        self.assertEqual(self.npc.consent_level, "safe")

    def test_reputation(self):
        homebrew_dialogue(self.npc, "You’re awesome", self.world)
        self.assertGreater(self.npc.reputation, 0)
        homebrew_dialogue(self.npc, "I hate you", self.world)
        self.assertLess(self.npc.reputation, 0.1)

    def test_memory_persistence(self):
        self.npc.memory.append(MemoryEntry(time.time(), "Test", "Reply", 0.5))
        self.npc.save_memory()
        new_npc = NPC(
            npc_id="npc_001",
            personality={"bravery": 0.3, "kindness": 0.7, "flirtiness": 0.6},
            routine={8: "work", 12: "lunch", 18: "home"}
        )
        new_npc.load_memory()
        self.assertEqual(len(new_npc.memory), 1)
        self.assertEqual(new_npc.memory[0].player_text, "Test")

# Entry point
def main():
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        unittest.main(argv=sys.argv[:1])
    else:
        npc = NPC(
            npc_id="npc_001",
            personality={"bravery": 0.3, "kindness": 0.7, "flirtiness": 0.6},
            routine={8: "work", 12: "lunch", 18: "home"}
        )
        dialogue_loop(npc)

if __name__ == "__main__":
    main()
