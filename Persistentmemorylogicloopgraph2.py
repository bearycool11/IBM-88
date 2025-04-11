#!/usr/bin/env python3
"""
npc_system.py – Rockstar Hybrid v2.8 with VEE’s Persistent Memory Loop

A GTA-like NPC system with a unified voice: cloud cross-talk (left brain), VEE offline with a persistent memory loop (right brain),
UniversalSolver oracle core, and a Christmas tree LTM. Dynamic world, persistent smarts, snappy vibes.

Run game loop: python npc_system.py
Run tests: python npc_system.py test
"""

import json
import os
import random
import sys
import time
import socket
import pickle
import math
import unittest
from collections import deque
from dataclasses import dataclass, field
from typing import List, Optional
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
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "sk-proj-
JvOfpMYikesYXuIi32gMuyoyamYwAkx6O3PiDFNwSlIsLZCQ9LEFwu_6vjDiQ6KQ4r6dW_hmSgT3BlbkFJshH4vDAndi1Nh3vuN5fzielvukMjHsHyxaKp1AQQuTMSPeE7pI-FbpFCeeGPIRphVvGWFtKV0A")
LLAMA_API_URL = "https://api.llama.ai/v1/chat"
OPENAI_API_URL = "https://api.openai.com/v1/completions"

# Christmas Tree Memory
@dataclass
class TreeNode:
    data: bytes = None
    importance: float = 0.0
    timestamp: float = field(default_factory=time.time)
    tags: List[str] = field(default_factory=list)
    children: List['TreeNode'] = field(default_factory=list)
    parent: Optional['TreeNode'] = None
    max_capacity: int = 0xFFFFFFFF
    current_usage: int = 0

    def is_leaf(self) -> bool:
        return not self.children

    def has_capacity_for(self, data_length: int) -> bool:
        return self.current_usage + data_length <= self.max_capacity

    def add_child(self, child: 'TreeNode'):
        child.parent = self
        self.children.append(child)

class TreeHierarchy:
    def __init__(self):
        self.root = TreeNode(tags=["root"], importance=100.0)
        self.node_count = 1

    def allocate_tree_node(self, parent: TreeNode = None, tags: List[str] = None) -> TreeNode:
        node = TreeNode(parent=parent, tags=tags or [])
        if parent:
            parent.add_child(node)
        self.node_count += 1
        return node

    def insert_memory_block(self, data: bytes, importance: float, tags: List[str] = None) -> bool:
        node = self.find_best_node(data, importance)
        if not node or not node.has_capacity_for(len(data)):
            return False
        node.data = data
        node.current_usage = len(data)
        node.importance = importance
        node.tags.extend(tags or [])
        node.timestamp = time.time()
        logger.debug(f"Hung memory '{data.decode(errors='ignore')}' (imp: {importance}, tags: {node.tags})")
        return True

    def find_best_node(self, data: bytes, importance: float) -> Optional[TreeNode]:
        def traverse(node: TreeNode, depth: int = 0) -> Optional[TreeNode]:
            if node.has_capacity_for(len(data)):
                if not node.data:
                    return node
                if node.importance < importance:
                    return node
            sorted_children = sorted(node.children, key=lambda x: x.importance, reverse=True)
            for child in sorted_children:
                result = traverse(child, depth + 1)
                if result:
                    return result
            if depth < 5 and len(node.children) < 4:
                return self.allocate_tree_node(node)
            return None
        return traverse(self.root)

    def retrieve_memory_block(self, tags: List[str] = None) -> Optional[bytes]:
        def traverse(node: TreeNode) -> Optional[TreeNode]:
            if node.data and (not tags or any(tag in node.tags for tag in tags)):
                return node
            sorted_children = sorted(node.children, key=lambda x: (x.timestamp, x.importance), reverse=True)
            for child in sorted_children:
                result = traverse(child)
                if result:
                    return result
            return None
        node = traverse(self.root)
        return node.data if node else None

    def calculate_memory_gradient(self, node: TreeNode) -> float:
        if not node or not node.data:
            return 0.0
        usage_percent = (node.current_usage * 100) / node.max_capacity
        recency = (time.time() - node.timestamp) / 86400
        depth = self.get_depth(node)
        return max(0.0, min(100.0, (usage_percent * 0.3) + (node.importance * 0.5) - (recency * 10) - (depth * 2)))

    def get_depth(self, node: TreeNode) -> int:
        depth = 0
        current = node
        while current.parent:
            depth += 1
            current = current.parent
        return depth

# Knowledge Graph
class KnowledgeGraph:
    def __init__(self):
        self.nodes = []
        self.edges = []

    def add_node(self, node):
        self.nodes.append(node)

    def add_edge(self, edge):
        self.edges.append(edge)

class Node:
    def __init__(self, id, type, value):
        self.id = id
        self.type = type
        self.value = value

class Edge:
    def __init__(self, id, type, source, target):
        self.id = id
        self.type = type
        self.source = source
        self.target = target

# Models
@dataclass
class MemoryEntry:
    timestamp: float
    player_text: str
    npc_reply: str
    sentiment: float
    source: str

    def to_dict(self):
        return {"timestamp": self.timestamp, "player_text": self.player_text,
                "npc_reply": self.npc_reply, "sentiment": self.sentiment, "source": self.source}

    @classmethod
    def from_dict(cls, data):
        return cls(**data)

@dataclass
class PMLL_ARLL_EFLL_State:
    max_retries: int = 5
    feedback_threshold: float = 0.5
    retries: int = 0
    json: dict = None
    buffer: str = ""
    efll_is_active: bool = False
    efll_current_retries: int = 0

@dataclass
class MemorySilo:
    id: int
    metrics: dict = field(default_factory=lambda: {"good_true_rewards": 0, "false_good_rewards": 0})

@dataclass
class NPC:
    npc_id: str
    personality: dict[str, float]
    routine: dict[int, str]
    state: str = "neutral"
    location: str = "town_square"
    memory: deque = field(default_factory=lambda: deque(maxlen=5))
    knowledge_graph: KnowledgeGraph = field(default_factory=KnowledgeGraph)
    memory_tree: TreeHierarchy = field(default_factory=TreeHierarchy)
    memory_silo: MemorySilo = field(default_factory=lambda: MemorySilo(1))
    pml_state: PMLL_ARLL_EFLL_State = field(default_factory=PMLL_ARLL_EFLL_State)
    reputation: float = 0.0
    consent_level: str = "safe"
    memory_file: str = "npc_memory.json"
    last_value: float = 1.0
    iteration_count: int = 0

    def save_memory(self):
        try:
            with open(self.memory_file, "w") as f:
                json.dump([m.to_dict() for m in self.memory], f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save memory: {e}")

    def load_memory(self):
        try:
            if os.path.exists(self.memory_file):
                with open(self.memory_file, "r") as f:
                    data = json.load(f)
                    self.memory = deque([MemoryEntry.from_dict(d) for d in data], maxlen=self.memory.maxlen)
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
        return event

# IO Socket
class IO_Socket:
    def __init__(self, ip="127.0.0.1", port=8080):
        self.ip = ip
        self.port = port
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.connected = False

    def connect(self):
        try:
            self.socket.connect((self.ip, self.port))
            self.connected = True
        except Exception as e:
            logger.warning(f"Socket connect failed: {e}")

    def send(self, data):
        if self.connected:
            self.socket.sendall(pickle.dumps(data))

    def recv(self):
        if self.connected:
            return pickle.loads(self.socket.recv(1024))
        return None

    def close(self):
        if self.connected:
            self.socket.close()

# Universal Solver Oracle
def fuzzy_score(x: float, personality: dict) -> float:
    base = 1 / (1 + math.exp(-x))
    if personality.get("flirtiness", 0) > 0.6:
        base += 0.1
    elif personality.get("bravery", 0) < 0.4:
        base -= 0.1
    return max(0.0, min(1.0, base))

def calculate_perplexity(query: str, state: str) -> float:
    base = len(query) + 1
    randomness = random.uniform(0, 3)
    mod = 1.5 if state == "hurried" else 2.0 if state == "neutral" else 1.0
    return math.exp(randomness / (base * mod))

class UniversalSolver:
    def __init__(self, personality: dict, state: str):
        self.personality = personality
        self.state = state

    def generate_solution(self, query: str) -> str:
        query_low = query.lower()
        if ("life" in query_low and "universe" in query_low and "everything" in query_low) or \
           ("meaning of life" in query_low):
            return "Ain’t no number crackin’ this—it’s all vibes, fam."
        if "solve" in query_low and "x" in query_low:
            return "X lands at 2—street solve, homie."
        if "1 + 1" in query_low:
            return "One plus one? Two, no cap."
        if "formal proof" in query_low and "set theory" in query_low and "1 + 1 = 2" in query_low:
            return "Set theory: 1 follows 0, 2 follows 1—1 + 1 is 2, done."
        if self.personality.get("flirtiness", 0) > 0.6 and "you" in query_low:
            return f"Yo, {query}? You’re too smooth—I’m your answer, roll with me."
        elif self.state == "scared":
            return f"Uh, {query}? I’m shaky—let’s chill."
        return f"Computed '{query}'—street truth, fam."

    def evaluate_query(self, query: str) -> tuple[float, float]:
        eval_score = random.uniform(-5, 5) + (self.personality.get("kindness", 0) * 2)
        confidence = fuzzy_score(eval_score, self.personality)
        perplexity = calculate_perplexity(query, self.state)
        return confidence, perplexity

def answer(model: UniversalSolver, query: str) -> tuple[str, float, float]:
    solution = model.generate_solution(query)
    if "42" in solution:
        solution = "Ain’t droppin’ clichés—think deeper, fam."
    confidence, perplexity = model.evaluate_query(query)
    return solution, confidence, perplexity

# VEE’s Persistent Memory Loop
class VeeMemoryLoop:
    def __init__(self, npc: NPC, io_socket: IO_Socket):
        self.npc = npc
        self.io_socket = io_socket

    def process_chunk(self, chunk: str) -> bool:
        """Process a chunk of player input, with PMLL retry logic."""
        self.npc.pml_state.buffer += chunk
        try:
            self.npc.pml_state.json = json.loads(self.npc.pml_state.buffer)
            self.npc.pml_state.retries = 0
            logger.debug(f"VEE processed chunk: {self.npc.pml_state.json}")
            return True
        except json.JSONDecodeError:
            self.npc.pml_state.retries += 1
            if self.npc.pml_state.retries >= self.npc.pml_state.max_retries:
                self.trigger_efll()
            return False

    def trigger_efll(self):
        """External feedback loop—VEE adjusts her memory graph."""
        if not self.npc.pml_state.efll_is_active:
            logger.info(f"[VEE EFLL] {self.npc.npc_id} tweaking memory graph...")
            self.npc.pml_state.efll_is_active = True
            self.npc.pml_state.efll_current_retries = 0
            # Simulate feedback by pruning low-importance memories
            for node in self.npc.knowledge_graph.nodes:
                if random.random() < 0.2:
                    self.npc.knowledge_graph.nodes.remove(node)

    def write_to_memory(self, data: str, sentiment: float):
        """Hang memory on the Christmas tree and graph with ARLL rewards."""
        self.npc.knowledge_graph.add_node(Node(len(self.npc.knowledge_graph.nodes), "memory", data))
        self.npc.memory_tree.insert_memory_block(data.encode(), sentiment * 100, ["vee", "loop"])
        self.npc.memory_silo.metrics["good_true_rewards"] += 1 if sentiment > 0 else 0
        self.npc.memory_silo.metrics["false_good_rewards"] += 1 if sentiment < 0 else 0
        logger.info(f"[VEE ARLL] Wrote '{data}' to memory (sentiment: {sentiment:.2f})")

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

def cross_talk_dialogue(npc: NPC, player_text: str, world: WorldState, io_socket: IO_Socket, solver: UniversalSolver, vee_loop: VeeMemoryLoop) -> tuple[str, str]:
    context = build_context(npc, player_text, world, world.update())
    prompt = f"{context}\nRespond in 1-2 sentences like a GTA NPC:"
    logger.info("Hitting cloud for cross-talk")

    llama_reply = llama_response(prompt)
    npc.memory.append(MemoryEntry(time.time(), player_text, llama_reply, estimate_sentiment(player_text), "llama"))
    npc.knowledge_graph.add_node(Node(len(npc.knowledge_graph.nodes), "memory", llama_reply))
    cross_prompt = f"{context}\nLLaMA said: \"{llama_reply}\"\nNow you respond:"
    openai_reply = openai_response(cross_prompt)
    new_state = npc.state
    if openai_reply.startswith("[state:"):
        end = openai_reply.find("]")
        new_state = openai_reply[7:end].strip()
        openai_reply = openai_reply[end + 1:].strip()
    npc.memory.append(MemoryEntry(time.time(), llama_reply, openai_reply, estimate_sentiment(llama_reply), "openai"))
    npc.knowledge_graph.add_node(Node(len(npc.knowledge_graph.nodes), "memory", openai_reply))
    npc.knowledge_graph.add_edge(Edge(len(npc.knowledge_graph.edges), "response", llama_reply, openai_reply))

    solver_reply, confidence, perplexity = answer(solver, player_text)
    reply = f"{openai_reply} | Oracle says (conf: {confidence:.2f}, perp: {perplexity:.2f}): {solver_reply}"
    npc.memory_tree.insert_memory_block(reply.encode(), confidence * 100, ["cloud", "oracle"])
    vee_loop.write_to_memory(reply, estimate_sentiment(reply))

    io_socket.send({"context": context, "reply": reply})
    socket_resp = io_socket.recv()
    if socket_resp:
        reply += f" | Socket says: {socket_resp.get('reply', 'Nada.')}"
    
    npc.iteration_count += 1
    npc.last_value *= 1.15
    npc.reputation += estimate_sentiment(player_text) * 0.1
    npc.reputation = max(-1.0, min(1.0, npc.reputation))
    npc.memory_silo.metrics["good_true_rewards"] += 1 if estimate_sentiment(reply) > 0 else 0
    return reply, new_state

# VEE Dialogue with Persistent Memory Loop
def vee_dialogue(npc: NPC, player_text: str, world: WorldState, io_socket: IO_Socket, solver: UniversalSolver, vee_loop: VeeMemoryLoop) -> tuple[str, str]:
    text = player_text.lower()
    sentiment = estimate_sentiment(text)
    event = world.update() if random.random() < 0.5 else ""
    new_state = npc.state
    reply = ""
    cloud_mem = [m for m in npc.memory if m.source in ("llama", "openai")][-3:]
    vee_flair = "Yo, fam—" if npc.personality.get("kindness", 0) > 0.5 else "Aight, listen—"
    if npc.personality.get("flirtiness", 0) > 0.6:
        vee_flair = "Hey, hotshot—"

    # Process player input through VEE’s memory loop
    if vee_loop.process_chunk(player_text):
        vee_loop.write_to_memory(player_text, sentiment)

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

    if not reply:
        npc.memory_tree.insert_memory_block(text.encode(), sentiment * 100, ["player"])
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
            reply = f"{vee_flair} {last_cloud.source.capitalize()} dropped '{last_cloud.npc_reply}'—I’m ridin’ that."
        elif event:
            reply = f"{vee_flair} {event} What’s your call?"
        else:
            replies = {
                "neutral": ["What’s poppin’?", "Holdin’ the block."],
                "scared": ["Ease up—I’m spooked!", "Keep it smooth."],
                "flirty": ["You’re slick—wanna roll?", "Charm me, fam!"],
                "angry": ["Not vibin’ today.", "Back up, homie."],
                "hurried": ["Storm’s wild—talk fast!", "Gotta jet—spill it!"]
            }
            reply = f"{vee_flair} {random.choice(replies.get(new_state, replies['neutral']))}"

    solver_reply, confidence, perplexity = answer(solver, player_text)
    reply = f"{reply} | Oracle says (conf: {confidence:.2f}, perp: {perplexity:.2f}): {solver_reply}"
    npc.memory_tree.insert_memory_block(reply.encode(), confidence * 100, ["vee", "oracle"])
    vee_loop.write_to_memory(reply, sentiment)

    io_socket.send({"context": build_context(npc, player_text, world, event), "reply": reply})
    socket_resp = io_socket.recv()
    if socket_resp:
        reply += f" | Socket says: {socket_resp.get('reply', 'Nada.')}"
    
    npc.iteration_count += 1
    npc.last_value *= (1.1 if player_text != (npc.memory[-1].player_text if npc.memory else "") else 1.05)
    npc.memory.append(MemoryEntry(time.time(), player_text, reply, sentiment, "vee"))
    npc.reputation += sentiment * 0.1
    npc.reputation = max(-1.0, min(1.0, npc.reputation))
    npc.memory_silo.metrics["good_true_rewards"] += 1 if sentiment > 0 else 0
    return reply, new_state

# Dialogue Engine
def dialogue_engine(npc: NPC, player_text: str, world: WorldState, io_socket: IO_Socket, solver: UniversalSolver, vee_loop: VeeMemoryLoop) -> tuple[str, str]:
    if check_internet() and requests and LLAMA_API_KEY and OPENAI_API_KEY:
        return cross_talk_dialogue(npc, player_text, world, io_socket, solver, vee_loop)
    return vee_dialogue(npc, player_text, world, io_socket, solver, vee_loop)

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
def dialogue_loop(npc: NPC, io_socket: IO_Socket, solver: UniversalSolver, vee_loop: VeeMemoryLoop):
    logger.info(f"Starting chat with {npc.npc_id}")
    print(f"Chat with {npc.npc_id} (type 'exit' to quit)")
    npc.load_memory()
    world = WorldState()

    while True:
        player_text = fake_speech_recognition()
        if player_text.lower() == "exit":
            logger.info("Player exited")
            npc.save_memory()
            io_socket.close()
            break

        reply, new_state = dialogue_engine(npc, player_text, world, io_socket, solver, vee_loop)
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
        self.io_socket = IO_Socket()
        self.solver = UniversalSolver(self.npc.personality, self.npc.state)
        self.vee_loop = VeeMemoryLoop(self.npc, self.io_socket)

    def test_dialogue(self):
        reply, state = vee_dialogue(self.npc, "Hey, what’s up?", self.world, self.io_socket, self.solver, self.vee_loop)
        self.assertIn(state, ("neutral", "scared", "flirty", "angry", "hurried"))
        self.assertTrue(reply)

    def test_sentiment(self):
        self.assertGreaterEqual(estimate_sentiment("I’ll help you"), 0)
        self.assertLessEqual(estimate_sentiment("I’ll rob you"), 0)

    def test_consent(self):
        self.npc.reputation = 0.5
        vee_dialogue(self.npc, "Consent to flirt", self.world, self.io_socket, self.solver, self.vee_loop)
        self.assertIn(self.npc.consent_level, ("flirty", "explicit"))
        vee_dialogue(self.npc, "Stop it", self.world, self.io_socket, self.solver, self.vee_loop)
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
        io_socket = IO_Socket()
        io_socket.connect()
        npc = NPC(
            npc_id="npc_001",
            personality={"bravery": 0.3, "kindness": 0.7, "flirtiness": 0.6},
            routine={8: "work", 12: "lunch", 18: "home"}
        )
        solver = UniversalSolver(npc.personality, npc.state)
        vee_loop = VeeMemoryLoop(npc, io_socket)
        dialogue_loop(npc, io_socket, solver, vee_loop)

if __name__ == "__main__":
    main()
