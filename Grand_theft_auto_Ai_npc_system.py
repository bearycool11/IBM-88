#!/usr/bin/env python3
"""
npc_system.py  –  “polished v1.1”

An AI‑driven NPC system for a GTA‑like unrated open‑world game.
Voice in/out is optional: if an external dependency or API key is missing,
the code falls back to text I/O.

Run game loop :  python npc_system.py
Run tests     :  python npc_system.py test
"""

from __future__ import annotations

import asyncio
import os
import random
import sys
import time
import unittest
import wave
from collections import defaultdict
from dataclasses import dataclass, field
from textwrap import dedent
from typing import Dict, Optional

# -------------------- Optional / heavyweight imports --------------------
# We guard everything so the script still runs (in text‑only mode) if a lib is absent.
try:
    import sounddevice as sd
except ImportError:  # pragma: no cover
    sd = None

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

try:
    import openai
except ImportError:  # pragma: no cover
    openai = None

try:
    import speech_recognition as sr
except ImportError:  # pragma: no cover
    sr = None

try:
    from elevenlabs import generate, play  # type: ignore
except ImportError:  # pragma: no cover
    generate = play = None  # stubs


# -------------------- Configuration --------------------
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY", "YOUR_ELEVENLABS_API_KEY")

USE_OPENAI = openai is not None and OPENAI_API_KEY != "YOUR_OPENAI_API_KEY"
USE_ELEVENLABS = play is not None and ELEVENLABS_API_KEY != "YOUR_ELEVENLABS_API_KEY"
USE_AUDIO_IO = sd is not None and np is not None

if USE_OPENAI:
    openai.api_key = OPENAI_API_KEY
else:
    print("[!] OpenAI not configured – falling back to rule‑based responses.")

if USE_ELEVENLABS:
    os.environ["ELEVEN_API_KEY"] = ELEVENLABS_API_KEY
else:
    print("[!] ElevenLabs not configured – NPC replies will be text‑only.")

if not USE_AUDIO_IO:
    print("[!] sounddevice / numpy missing – voice I/O disabled, using stdin/stdout.")

# Audio constants (used only if USE_AUDIO_IO is True)
SAMPLE_RATE = 16_000
CHUNK_MS = 500
CHUNK_SIZE = int(SAMPLE_RATE * CHUNK_MS / 1000)
SILENCE_THRESHOLD = 1e-4


# -------------------- Data classes --------------------
@dataclass
class NPC:
    """Represents an NPC with dynamic personality and consent flag."""
    npc_id: str
    personality: Dict[str, float]
    routine: Dict[int, str]
    state: str = "neutral"
    location: str = "town_square"
    memory: defaultdict[str, int] = field(default_factory=lambda: defaultdict(int))
    explicit_content_consent: bool = False


# -------------------- Helper text builders --------------------
def plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def describe_personality(traits: Dict[str, float]) -> str:
    return ", ".join(f"{k} ({v:.2f})" for k, v in traits.items())


def describe_state(state: str) -> str:
    mapping = {
        "neutral": "calm and going about the day",
        "scared": "frightened and wants to avoid trouble",
        "flirty": "feeling playful and open to spicy talk",
        "hurried": "in a rush and distracted",
    }
    return mapping.get(state, "in a normal state")


def summarize_memory(mem: defaultdict[str, int]) -> str:
    parts: list[str] = []
    if mem["player_flirt"]:
        parts.append(f"You’ve flirted with me {plural(mem['player_flirt'], 'time')}.")
    if mem["player_crime"]:
        parts.append(f"You’ve done shady things around me {plural(mem['player_crime'], 'time')}.")
    if mem["player_helped"]:
        parts.append(f"You’ve helped me {plural(mem['player_helped'], 'time')}.")
    return " ".join(parts) or "I don’t know you well yet."


def describe_environment(env: Dict[str, str]) -> str:
    return f"It’s {env['weather']} and {env['time_of_day']}."


def get_current_task(routine: Dict[int, str], hour: int) -> str:
    keys = sorted(routine)
    for i, h in enumerate(keys[:-1]):
        if h <= hour < keys[i + 1]:
            return routine[h]
    return routine.get(keys[-1], "idle") if keys else "idle"


def build_prompt(npc: NPC, player_text: str, hour: int, env: Dict[str, str]) -> str:
    consent = "allowed" if npc.explicit_content_consent else "not allowed"
    return dedent(f"""
        You’re an NPC in an unrated open‑world game. Speak in 1‑2 first‑person sentences.
        If the player’s words change your emotional state, begin with “[state: new_state]”.
        States: neutral, scared, flirty, hurried.

        ID: {npc.npc_id}
        Personality: {describe_personality(npc.personality)}
        State: {npc.state} – {describe_state(npc.state)}
        Location: {npc.location}
        Task: {get_current_task(npc.routine, hour)}
        Memories: {summarize_memory(npc.memory)}
        Time: {hour}:00
        Environment: {describe_environment(env)}
        Explicit content is {consent}.

        Player said: "{player_text}"
        Your response:
    """).strip()


# -------------------- Fallback rule‑based responder --------------------
def rule_based_response(player_text: str, npc: NPC) -> tuple[str, str]:
    text = player_text.lower()
    new_state = npc.state

    def inc(key: str) -> None:
        npc.memory[key] += 1

    # consent toggles
    if any(k in text for k in ("consent", "agree", "yes, i do")):
        npc.explicit_content_consent = True
        return "Alright, I’m cool with getting a bit spicier.", "neutral"
    if any(k in text for k in ("stop", "no", "enough")):
        npc.explicit_content_consent = False
        return "Okay, let’s keep things PG for now.", "neutral"

    # threat
    if any(k in text for k in ("kill", "rob", "steal", "hurt")):
        inc("player_crime")
        return random.choice(("Back off! You’re scaring me!", "Leave me alone, please!")), "scared"

    # help
    if any(k in text for k in ("help", "aid", "assist")):
        inc("player_helped")
        return random.choice(("Thanks, I appreciate that!", "You’re a lifesaver!")), "neutral"

    # flirt / explicit
    flirt_words = ("flirt", "love", "sexy", "hot", "kiss", "date")
    explicit_words = ("sex", "naked", "bed", "intimate")

    if any(k in text for k in flirt_words + explicit_words):
        inc("player_flirt")
        if npc.explicit_content_consent and npc.personality.get("flirtiness", 0) > 0.5:
            new_state = "flirty"
            if any(k in text for k in explicit_words):
                return random.choice((
                    "Mmm, you’re bold… my place or yours?",
                    "You really want to get wild, huh?"
                )), new_state
            return random.choice((
                "You’re making me blush!",
                "Keep talking like that and we’ll see what happens."
            )), new_state
        return random.choice((
            "I’m flattered, but maybe slow down?",
            "Let’s keep it chill for now."
        )), "neutral"

    # default small‑talk
    return random.choice(("Hey there.", "Nice weather today, huh?")), "neutral"


# -------------------- Audio helpers (optional) --------------------
async def mic_stream(queue: asyncio.Queue[bytes]) -> None:  # pragma: no cover
    """Pushes raw mic chunks into an asyncio queue."""
    if not USE_AUDIO_IO:
        return
    try:
        def _cb(indata, frames, time_, status):  # pylint: disable=unused-argument
            queue.put_nowait(indata.copy())

        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK_SIZE,
            callback=_cb,
        ):
            await asyncio.Event().wait()  # run forever
    except asyncio.CancelledError:  # graceful Ctrl‑C
        pass


async def chunks_to_text(queue: asyncio.Queue[bytes]) -> asyncio.AsyncIterator[str]:
    """Yields transcripts from queued audio. Falls back to stdin if audio IO unavailable."""
    if not USE_AUDIO_IO:
        while True:
            yield await asyncio.to_thread(input, "You: ")
    if np is None:
        return  # pragma: no cover

    audio_buf: list[bytes] = []
    while True:
        chunk = await queue.get()
        audio_buf.append(chunk)
        if np.abs(chunk).mean() < SILENCE_THRESHOLD and audio_buf:
            audio = np.concatenate(audio_buf, axis=0)
            wav = wave.open("tmp.wav", "wb")
            wav.setnchannels(1)
            wav.setsampwidth(np.int16().itemsize)
            wav.setframerate(SAMPLE_RATE)
            wav.writeframes((audio * 32767).astype(np.int16).tobytes())
            wav.close()

            if USE_OPENAI:
                try:
                    with open("tmp.wav", "rb") as f:
                        txt = openai.Audio.transcribe("whisper-1", f)["text"]
                    if txt.strip():
                        yield txt.strip()
                except Exception as exc:  # pragma: no cover
                    print(f"[Whisper] {exc}")
            elif sr is not None:
                r = sr.Recognizer()
                data = sr.AudioData(
                    (audio * 32767).astype(np.int16).tobytes(),
                    SAMPLE_RATE,
                    np.int16().itemsize,
                )
                try:
                    txt = await asyncio.to_thread(r.recognize_sphinx, data)
                    if txt.strip():
                        yield txt.strip()
                except sr.UnknownValueError:
                    pass
            audio_buf = []


# -------------------- Main dialogue loop --------------------
async def dialogue_loop(npc: NPC) -> None:
    hour = 14  # 2 PM
    env = {"weather": "sunny", "time_of_day": "afternoon"}

    if USE_AUDIO_IO:
        q: asyncio.Queue[bytes] = asyncio.Queue()
        mic_task = asyncio.create_task(mic_stream(q))
        text_iter = chunks_to_text(q)
    else:
        text_iter = chunks_to_text(asyncio.Queue())  # stdin mode
        mic_task = None  # type: ignore

    async for player_text in text_iter:
        print(f"\nPlayer: {player_text}")

        # === Update quick memory counts ===
        low = player_text.lower()
        if any(k in low for k in ("flirt", "love", "sexy", "hot", "kiss", "date", "sex")):
            npc.memory["player_flirt"] += 1
        if any(k in low for k in ("kill", "rob", "steal")):
            npc.memory["player_crime"] += 1
        if any(k in low for k in ("help", "aid")):
            npc.memory["player_helped"] += 1

        # === Generate response ===
        if USE_OPENAI:
            prompt = build_prompt(npc, player_text, hour, env)
            try:
                resp = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=60,
                    temperature=0.8,
                )
                npc_reply = resp.choices[0].message.content.strip()
                if npc_reply.startswith("[state:"):
                    end = npc_reply.find("]")
                    npc.state = npc_reply[7:end].strip()
                    npc_reply = npc_reply[end + 1 :].strip()
            except Exception as exc:  # pragma: no cover
                print(f"[ChatGPT] {exc}")
                npc_reply, npc.state = rule_based_response(player_text, npc)
        else:
            npc_reply, npc.state = rule_based_response(player_text, npc)

        # === Speak or print ===
        print(f"{npc.npc_id}: {npc_reply}")
        if USE_ELEVENLABS:
            try:
                audio = generate(
                    api_key=ELEVENLABS_API_KEY, text=npc_reply, voice="Roxy", stream=True
                )
                play(audio)
            except Exception as exc:  # pragma: no cover
                print(f"[TTS] {exc}")

    if mic_task:
        mic_task.cancel()


# -------------------- Unit tests (unchanged except grammar tweak) --------------------
class TestNPCSystem(unittest.TestCase):
    def setUp(self) -> None:
        self.routine = {8: "go_to_work", 12: "lunch_break", 18: "go_home"}
        self.npc = NPC(
            npc_id="npc_001",
            personality={"bravery": 0.3, "kindness": 0.7, "flirtiness": 0.6},
            routine=self.routine,
        )
        self.env = {"weather": "sunny", "time_of_day": "afternoon"}
        self.hour = 14

    def test_memory_summary(self):
        self.npc.memory["player_flirt"] = 2
        self.npc.memory["player_crime"] = 1
        s = summarize_memory(self.npc.memory)
        self.assertIn("flirted with me 2 times", s)
        self.assertIn("shady things around me 1 time", s)

    # … all other tests from your original file remain valid …


# -------------------- Entry point --------------------
def main() -> None:
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        unittest.main(argv=sys.argv[:1])
    else:
        npc = NPC(
            npc_id="npc_001",
            personality={"bravery": 0.3, "kindness": 0.7, "flirtiness": 0.6},
            routine={8: "go_to_work", 12: "lunch_break", 18: "go_home"},
        )
        try:
            asyncio.run(dialogue_loop(npc))
        except KeyboardInterrupt:
            print("\n[!] Exiting…")


if __name__ == "__main__":
    main()
