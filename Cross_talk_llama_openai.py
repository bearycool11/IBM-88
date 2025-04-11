#!/usr/bin/env python3
"""
npc_system.py – Rockstar Hybrid v2.5 with Cloud Cross-Talk & VEE/VER

An AI-driven NPC system for a GTA-like open-world game, built from scratch.
LLaMA and OpenAI cross-talk online; VEE/VER riffs on memory offline.
Dynamic world, persistent memory, snappy vibes.

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

try:
    import requests
except ImportError:
    requests = None

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("pmll_cross_talk.log")]
)
logger = logging.getLogger(__name__)

# API Keys
LLAMA_API_KEY = os.getenv("LLAMA_API_KEY", "LA-8c4003a74c5040b2b735866f22e754ed55c2ab712b0346b3bca0f1993362704a")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-proj
JvOfpMYikesYXuIi32gMuyoyamYwAkx6O3PiDFNwSlIsLZCQ9LEFwu_6vjDiQ6KQ4r6dW_hmSgT3BlbkFJshH4vDAndi1Nh3vuN5fzielvukMjHsHyxaKp1AQQuTMSPeE7pI-FbpFCeeGPIRphVvGWFtKV0A")
LLAMA_API_URL = "https://api.llama.ai/v1/chat"
OPENAI_API_URL = "https://api.openai.com/v1/completions"

# Models
@dataclass
class MemoryEntry:
    timestamp: float
    player_text: str
    npc_reply: str
    sentiment: float
    source: str  # "player", "llama", "openai", "vee"

    def to_dict(self):
        return {"timestamp": self.timestamp, "player_text": self.player_text,
                "npc_reply": self.npc_reply, "sentiment": self.sentiment, "source": self.source}

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

@dataclass
class NPC:
    npc_id: str
    personality: dict[str, float]
    routine: dict[int, str]
    state: str = "neutral"
    location: str = "town_square"
    memory: deque = field(default_factory=lambda: deque(maxlen=5))
    reputation: float = 0.0
    consent_level: str = "safe"
    memory_file: str = "npc_memory.json"
    last_value: float = 1.0
    iteration_count: int = 0

    def save_memory(self):
        try:
            with open(self.memory_file, "w") as f:
                json.dump([m.to_dict() for m in self.memory], f, indent=2)
            logger.debug(f"Saved {len(self.memory)} memories for {self.npc_id}")
        except Exception as e:
            logger.warning(f"Failed to save memory: {e}")

    def load_memory(self):
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, "r") as f:
                    data = json.load(f)
                    self.memory = deque([MemoryEntry.from_dict(d) for d in data], maxlen=self.memory.maxlen)
                logger.info(f"Loaded {len(self.memory)} memories for {self.npc_id}")
        except Exception as e:
            logger.warning(f"Failed to load memory: {e}")

class WorldState:
    def __init__(self):
        self.hour = 14
        self.weather = "sunny"
        self.time_of_day = "afternoon"
        self.tick_interval = 30
        self.events = [
            "Fight broke out nearby!",
            "Cops chasing someone!",
            "Deal at the docks...",
            "Music blasting down the block."
        ]

    def update(self) -> str:
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

# Dialogue Helpers
def estimate_sentiment(text: str) -> float:
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

# Cloud Cross-Talk
def check_internet() -> bool:
    if not requests:
        return False
    try:
        requests.get("https://www.google.com", timeout=2)
        return True
    except (requests.RequestException, TimeoutError):
        logger.debug("No internet detected")
        return False

def llama_response(prompt: str) -> str:
    try:
        headers = {"Authorization": f"Bearer {LLAMA_API_KEY}", "Content-Type": "application/json"}
        data = {"prompt": prompt, "max_tokens": 60, "temperature": 0.8}
        resp = requests.post(LLAMA_API_URL, headers=headers, json=data, timeout=5)
        resp.raise_for_status()
        return resp.json().get("text", "Yo, I got nothing.").strip()
    except Exception as e:
        logger.error(f"LLaMA error: {e}")
        return "LLaMA’s down, fam."

def openai_response(prompt: str) -> str:
    try:
        headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
        data = {"prompt": prompt, "max_tokens": 60, "temperature": 0.9}
        resp = requests.post(OPENAI_API_URL, headers=headers, json=data, timeout=5)
        resp.raise_for_status()
        return resp.json()["choices"][0]["text"].strip()
    except Exception as e:
        logger.error(f"OpenAI error: {e}")
        return "OpenAI’s offline, yo."

def cross_talk_dialogue(npc: NPC, player_text: str, world: WorldState) -> tuple[str, str]:
    context = build_context(npc, player_text, world, world.update())
    prompt = f"{context}\nRespond in 1-2 sentences like a GTA NPC:"
    logger.info("Hitting cloud for cross-talk")

    llama_reply = llama_response(prompt)
    npc.memory.append(MemoryEntry(time.time(), player_text, llama_reply, estimate_sentiment(player_text), "llama"))
    cross_prompt = f"{context}\nLLaMA said: \"{llama_reply}\"\nNow you respond:"
    reply = openai_response(cross_prompt)
    new_state = npc.state
    if reply.startswith("[state:"):
        end = reply.find("]")
        new_state = reply[7:end].strip()
        reply = reply[end + 1:].strip()
    npc.memory.append(MemoryEntry(time.time(), llama_reply, reply, estimate_sentiment(llama_reply), "openai"))
    npc.iteration_count += 1
    npc.last_value *= 1.15
    npc.reputation += estimate_sentiment(player_text) * 0.1
    npc.reputation = max(-1.0, min(1.0, npc.reputation))
    logger.debug(f"Cloud: Iter {npc.iteration_count}, Value {npc.last_value:.2f}")
    return reply, new_state

# VEE/VER Dialogue
def vee_dialogue(npc: NPC, player_text: str, world: WorldState) -> tuple[str, str]:
    text = player_text.lower()
    sentiment = estimate_sentiment(text)
    event = world.update() if random.random() < 0.5 else ""
    new_state = npc.state
    reply = ""
    cloud_mem = [m for m in npc.memory if m.source in ("llama", "openai")][-3:]

    # VEE’s persona
    vee_flair = "Yo, fam—" if npc.personality.get("kindness", 0) > 0.5 else "Aight, listen—"
    if npc.personality.get("flirtiness", 0) > 0.6:
        vee_flair = "Hey, hotshot—"

    # Consent
    if "consent" in text:
        if npc.reputation > 0 and npc.personality.get("flirtiness", 0) > 0.4:
            npc.consent_level = "flirty"
            reply = f"{vee_flair} I’m down to flirt—hit me!"
            new_state = "flirty"
        elif npc.reputation > 0.5:
            npc.consent_level = "explicit"
            reply = f"{vee_flair} Let’s get wild, fam."
            new_state = "flirty"
        else:
            reply = f"{vee_flair} Nah, we’re cool."
    elif "stop" in text:
        npc.consent_level = "safe"
        reply = f"{vee_flair} Aight, keepin’ it chill."
        new_state = "neutral"

    # State transitions
    state_probs = {
        "neutral": 0.4 + npc.personality.get("kindness", 0) * 0.2,
        "scared": 0.3 if npc.reputation < 0 else 0.1,
        "flirty": 0.3 if npc.consent_level in ("flirty", "explicit") else 0.0,
        "angry": 0.2 if npc.reputation < -0.3 else 0.1,
        "hurried": 0.2 if world.weather == "stormy" else 0.1
    }
    if random.random() < 0.3:
        total = sum(state_probs.values())
        if total > 0:
            new_state = random.choices(list(state_probs.keys()), weights=[v / total for v in state_probs.values()])[0]

    # Response logic
    if not reply:
        if "kill" in text or "rob" in text:
            new_state = "scared"
            reply = f"{vee_flair} Back off—I ain’t messin’ with that!"
        elif "help" in text:
            reply = f"{vee_flair} Thanks, that’s real."
        elif any(k in text for k in ("love", "hot", "date")) and npc.consent_level != "safe":
            new_state = "flirty"
            reply = f"{vee_flair} You’re heatin’ the block—keep it comin’!"
        elif cloud_mem and random.random() < 0.5:
            last_cloud = random.choice(cloud_mem)
            reply = f"{vee_flair} {last_cloud.source.capitalize()} dropped '{last_cloud.npc_reply}'—I’m ridin’ that wave."
        elif event:
            reply = f"{vee_flair} {event} What’s your call?"
        else:
            replies = {
                "neutral": ["What’s poppin’, homie?", "Holdin’ it down here."],
                "scared": ["Ease up—I’m spooked!", "Keep it smooth, fam."],
                "flirty": ["You’re slick—wanna roll?", "Hit me with that charm!"],
                "angry": ["Not vibin’ today.", "Back up, bruh."],
                "hurried": ["Storm’s crazy—talk fast!", "Gotta bounce—spill it!"]
            }
            reply = f"{vee_flair} {random.choice(replies.get(new_state, replies['neutral']))}"

    npc.iteration_count += 1
    npc.last_value *= (1.1 if player_text != (npc.memory[-1].player_text if npc.memory else "") else 1.05)
    npc.memory.append(MemoryEntry(time.time(), player_text, reply, sentiment, "vee"))
    npc.reputation += sentiment * 0.1
    npc.reputation = max(-1.0, min(1.0, npc.reputation))
    logger.debug(f"VEE: Iter {npc.iteration_count}, Value {npc.last_value:.2f}")
    return reply, new_state

# Dialogue Engine
def dialogue_engine(npc: NPC, player_text: str, world: WorldState) -> tuple[str, str]:
    if check_internet() and requests and LLAMA_API_KEY and OPENAI_API_KEY:
        return cross_talk_dialogue(npc, player_text, world)
    return vee_dialogue(npc, player_text, world)

def fake_tts(text: str):
    print(f"🎙️ {text}")

def fake_speech_recognition() -> str:
    try:
        text = input("🎤 You: ").strip()
        if random.random() < 0.1:
            text = text.replace("you", "ya") or text + " uh"
        return text
    except KeyboardInterrupt:
        return "exit"

# Main Loop
def dialogue_loop(npc: NPC):
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

        reply, new_state = dialogue_engine(npc, player_text, world)
        npc.state = new_state
        print(f"\n{npc.npc_id} [{npc.state}, rep: {npc.reputation:.2f}, val: {npc.last_value:.2f}]:")
        fake_tts(reply)
        npc.save_memory()

# Tests
class TestNPCSystem(unittest.TestCase):
    def setUp(self):
        self.npc = NPC(
            npc_id="npc_001",
            personality={"bravery": 0.3, "kindness": 0.7, "flirtiness": 0.6},
            routine={8: "work", 12: "lunch", 18: "home"}
        )
        self.world = WorldState()

    def test_dialogue(self):
        reply, state = vee_dialogue(self.npc, "Hey, what’s up?", self.world)
        self.assertIn(state, ("neutral", "scared", "flirty", "angry", "hurried"))
        self.assertTrue(reply)

    def test_sentiment(self):
        self.assertGreaterEqual(estimate_sentiment("I’ll help you"), 0)
        self.assertLessEqual(estimate_sentiment("I’ll rob you"), 0)

    def test_consent(self):
        self.npc.reputation = 0.5
        vee_dialogue(self.npc, "Consent to flirt", self.world)
        self.assertIn(self.npc.consent_level, ("flirty", "explicit"))
        vee_dialogue(self.npc, "Stop it", self.world)
        self.assertEqual(self.npc.consent_level, "safe")

    def test_memory(self):
        self.npc.memory.append(MemoryEntry(time.time(), "Test", "Reply", 0.5, "vee"))
        self.npc.save_memory()
        new_npc = NPC(
            npc_id="npc_001",
            personality={"bravery": 0.3, "kindness": 0.7, "flirtiness": 0.6},
            routine={8: "work", 12: "lunch", 18: "home"}
        )
        new_npc.load_memory()
        self.assertEqual(len(new_npc.memory), 1)

# Entry Point
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
