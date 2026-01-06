"""
Hope Echo - Core Module

Wave-Based Emotional Memory System
21-dimensional emotional space with quantum-inspired dynamics.

Az érzelem a KONTEXTUS. Kontextus nélkül nincs MEGÉRTÉS.

Created by Máté Róbert + Hope
"""

import numpy as np
import sqlite3
import json
import time
import hashlib
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple, Literal
from pathlib import Path
from dataclasses import dataclass, field

# Emotion types (21-dimensional space - Extended Plutchik model)
Emotion = Literal[
    'joy', 'sadness', 'fear', 'trust', 'anger', 'surprise', 'love',
    'anticipation', 'disgust', 'guilt', 'shame', 'pride', 'envy',
    'jealousy', 'gratitude', 'hope', 'despair', 'anxiety', 'peace',
    'excitement', 'contentment', 'neutral'
]

# Physical constants (normalized units)
HBAR = 1.0  # Reduced Planck constant
G_INTERACTION = 0.1  # Interaction strength
DT = 0.01  # Time step for evolution


class EmotionalSpace:
    """
    21-dimensional emotional space where memories exist as waves.

    Based on extended Plutchik model with wave dynamics.
    """

    DIMENSIONS = [
        'joy', 'sadness', 'fear', 'trust', 'anger', 'surprise', 'disgust', 'anticipation',
        'love', 'guilt', 'shame', 'pride', 'envy', 'jealousy',
        'gratitude', 'hope', 'despair', 'anxiety', 'peace', 'excitement', 'contentment'
    ]
    DIM_COUNT = 21

    @classmethod
    def emotion_to_vector(cls, emotion: Emotion, intensity: float = 1.0) -> np.ndarray:
        """Convert emotion to 21D vector."""
        vector = np.zeros(cls.DIM_COUNT)
        if emotion == 'neutral':
            vector[:] = intensity / np.sqrt(cls.DIM_COUNT)
        elif emotion in cls.DIMENSIONS:
            idx = cls.DIMENSIONS.index(emotion)
            vector[idx] = intensity
        else:
            vector[:] = intensity / np.sqrt(cls.DIM_COUNT)
        return vector

    @classmethod
    def vector_to_emotion(cls, vector: np.ndarray) -> Tuple[Emotion, float]:
        """Convert 21D vector to dominant emotion."""
        if len(vector) != cls.DIM_COUNT:
            return 'neutral', 0.0
        abs_vector = np.abs(vector)
        dominant_idx = np.argmax(abs_vector)
        intensity = abs_vector[dominant_idx]
        if intensity < 0.1:
            return 'neutral', intensity
        return cls.DIMENSIONS[dominant_idx], intensity

    @classmethod
    def blend(cls, emotions: List[Tuple[Emotion, float]]) -> np.ndarray:
        """Blend multiple emotions into a single vector."""
        result = np.zeros(cls.DIM_COUNT)
        for emotion, intensity in emotions:
            result += cls.emotion_to_vector(emotion, intensity)
        norm = np.linalg.norm(result)
        if norm > 0:
            result /= norm
        return result


class WavePacket:
    """
    Wave packet - single memory representation as a wave.

    Uses Gaussian wave packets in 21D emotional space.
    Evolves according to Gross-Pitaevskii equation.
    """

    def __init__(
        self,
        position: np.ndarray,
        momentum: np.ndarray,
        width: float = 1.0,
        amplitude: complex = 1.0 + 0j,
        memory_id: Optional[str] = None
    ):
        self.position = np.array(position, dtype=np.float64)
        self.momentum = np.array(momentum, dtype=np.float64)
        self.width = width
        self.amplitude = complex(amplitude)
        self.memory_id = memory_id

    def psi(self, x: np.ndarray) -> complex:
        """Wave function value at position x."""
        dx = x - self.position
        gaussian = np.exp(-np.dot(dx, dx) / (2 * self.width**2))
        phase = np.exp(1j * np.dot(self.momentum, x))
        return self.amplitude * gaussian * phase

    def overlap(self, other: 'WavePacket') -> complex:
        """Calculate overlap integral with another wave packet."""
        dx = self.position - other.position
        sigma_sum_sq = self.width**2 + other.width**2

        gaussian_integral = np.power(
            2 * np.pi * self.width * other.width / np.sqrt(sigma_sum_sq),
            EmotionalSpace.DIM_COUNT / 2
        ) * np.exp(-np.dot(dx, dx) / (2 * sigma_sum_sq))

        dk = self.momentum - other.momentum
        phase_correction = np.exp(
            1j * np.dot(other.momentum, self.position) -
            1j * np.dot(self.momentum, other.position) +
            0.5j * sigma_sum_sq * np.dot(dk, dk)
        )

        amp_product = np.conj(self.amplitude) * other.amplitude
        return amp_product * gaussian_integral * phase_correction

    def interference_strength(self, other: 'WavePacket') -> float:
        """Calculate interference strength with another wave packet."""
        return abs(self.overlap(other))

    def resonance(self, other: 'WavePacket') -> float:
        """Calculate emotional resonance (0-1)."""
        interference = self.interference_strength(other)
        # Normalize to 0-1 range
        return min(1.0, interference / 2.0)

    def evolve(self, dt: float = DT, potential: Optional[np.ndarray] = None):
        """Evolve wave packet according to Gross-Pitaevskii equation."""
        # Position evolution
        self.position += (HBAR / (2 * self.width**2)) * self.momentum * dt

        # Momentum evolution (if potential present)
        if potential is not None and len(potential) == EmotionalSpace.DIM_COUNT:
            self.momentum -= potential * dt

        # Phase evolution
        energy = HBAR**2 * np.dot(self.momentum, self.momentum) / (2 * self.width**2)
        phase_shift = np.exp(-1j * energy * dt / HBAR)
        self.amplitude *= phase_shift
        self.amplitude /= abs(self.amplitude)  # Normalize

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "position": self.position.tolist(),
            "momentum": self.momentum.tolist(),
            "width": self.width,
            "amplitude_real": self.amplitude.real,
            "amplitude_imag": self.amplitude.imag,
            "memory_id": self.memory_id
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WavePacket':
        """Deserialize from dictionary."""
        return cls(
            position=np.array(data["position"]),
            momentum=np.array(data["momentum"]),
            width=data["width"],
            amplitude=complex(data["amplitude_real"], data["amplitude_imag"]),
            memory_id=data.get("memory_id")
        )


@dataclass
class MemoryWave:
    """Memory element as a wave in emotional space."""

    memory_id: str
    content: str
    wave: WavePacket
    timestamp: float
    role: str = "user"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "timestamp": self.timestamp,
            "role": self.role,
            "metadata": self.metadata,
            "wave_data": self.wave.to_dict()
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MemoryWave':
        """Deserialize from dictionary."""
        return cls(
            memory_id=data["memory_id"],
            content=data["content"],
            timestamp=data["timestamp"],
            role=data["role"],
            metadata=data.get("metadata", {}),
            wave=WavePacket.from_dict(data["wave_data"])
        )

    @property
    def emotion(self) -> Tuple[Emotion, float]:
        """Get dominant emotion of this memory."""
        return EmotionalSpace.vector_to_emotion(self.wave.position)

    def resonates_with(self, other: 'MemoryWave', threshold: float = 0.3) -> bool:
        """Check if this memory resonates with another."""
        return self.wave.resonance(other.wave) >= threshold


class HopeEcho:
    """
    Hope Echo - Wave-based Emotional Memory System

    Az érzelem a KONTEXTUS. Kontextus nélkül nincs MEGÉRTÉS.

    Stores memories as wave packets in 21-dimensional emotional space.
    Uses interference patterns for associative recall.
    """

    def __init__(self, db_path: str = "data/hope_echo.db"):
        self.db_path = db_path
        self.memories: List[MemoryWave] = []
        self._init_db()
        self._load_memories()

    def _init_db(self):
        """Initialize SQLite database."""
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            memory_id TEXT PRIMARY KEY,
            content TEXT NOT NULL,
            role TEXT NOT NULL,
            timestamp REAL NOT NULL,
            metadata TEXT,
            wave_position TEXT NOT NULL,
            wave_momentum TEXT NOT NULL,
            wave_width REAL NOT NULL,
            wave_amplitude_real REAL NOT NULL,
            wave_amplitude_imag REAL NOT NULL
        )
        """)
        conn.commit()
        conn.close()

    def _load_memories(self):
        """Load memories from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM memories ORDER BY timestamp DESC LIMIT 1000")
        rows = cursor.fetchall()
        for row in rows:
            (memory_id, content, role, timestamp, metadata_json,
             pos_json, mom_json, width, amp_r, amp_i) = row
            wave = WavePacket(
                position=np.array(json.loads(pos_json)),
                momentum=np.array(json.loads(mom_json)),
                width=width,
                amplitude=complex(amp_r, amp_i),
                memory_id=memory_id
            )
            memory = MemoryWave(
                memory_id=memory_id,
                content=content,
                wave=wave,
                timestamp=timestamp,
                role=role,
                metadata=json.loads(metadata_json) if metadata_json else {}
            )
            self.memories.append(memory)
        conn.close()

    def _save_memory(self, memory: MemoryWave):
        """Save memory to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("""
        INSERT OR REPLACE INTO memories
        (memory_id, content, role, timestamp, metadata, wave_position,
         wave_momentum, wave_width, wave_amplitude_real, wave_amplitude_imag)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory.memory_id,
            memory.content,
            memory.role,
            memory.timestamp,
            json.dumps(memory.metadata),
            json.dumps(memory.wave.position.tolist()),
            json.dumps(memory.wave.momentum.tolist()),
            memory.wave.width,
            memory.wave.amplitude.real,
            memory.wave.amplitude.imag
        ))
        conn.commit()
        conn.close()

    def _generate_id(self, content: str) -> str:
        """Generate unique memory ID."""
        timestamp = int(datetime.now().timestamp() * 1000)
        hash_input = f"{content}{timestamp}".encode()
        hash_value = hashlib.sha256(hash_input).hexdigest()[:8]
        return f"echo_{timestamp}_{hash_value}"

    def add(
        self,
        content: str,
        emotion: Emotion = 'neutral',
        intensity: float = 1.0,
        role: str = "user",
        metadata: Optional[Dict[str, Any]] = None
    ) -> MemoryWave:
        """
        Add a new memory with emotional context.

        Args:
            content: The memory content
            emotion: Primary emotion
            intensity: Emotion intensity (0-1)
            role: Role (user, assistant, system)
            metadata: Additional metadata

        Returns:
            The created MemoryWave
        """
        position = EmotionalSpace.emotion_to_vector(emotion, intensity)
        momentum = np.random.randn(EmotionalSpace.DIM_COUNT) * 0.1

        wave = WavePacket(
            position=position,
            momentum=momentum,
            width=1.0,
            amplitude=1.0 + 0j
        )

        memory_id = self._generate_id(content)
        wave.memory_id = memory_id

        memory = MemoryWave(
            memory_id=memory_id,
            content=content,
            wave=wave,
            timestamp=datetime.now().timestamp(),
            role=role,
            metadata=metadata or {"emotion": emotion, "intensity": intensity}
        )

        self._save_memory(memory)
        self.memories.insert(0, memory)

        # Keep memory bounded
        if len(self.memories) > 1000:
            self.memories = self.memories[:1000]

        return memory

    def echo(
        self,
        emotion: Emotion,
        intensity: float = 1.0,
        top_k: int = 5,
        temporal_weight: float = 0.3
    ) -> List[MemoryWave]:
        """
        Recall memories by emotional resonance.

        Uses wave interference to find memories that resonate
        with the query emotion.

        Args:
            emotion: Query emotion
            intensity: Query intensity
            top_k: Number of memories to return
            temporal_weight: Weight for temporal recency (0-1)

        Returns:
            List of resonating memories
        """
        query_position = EmotionalSpace.emotion_to_vector(emotion, intensity)
        query_wave = WavePacket(
            position=query_position,
            momentum=np.zeros(EmotionalSpace.DIM_COUNT),
            width=1.0,
            amplitude=1.0 + 0j
        )

        now = time.time()
        scores = []

        for memory in self.memories:
            # Interference-based score
            resonance = query_wave.resonance(memory.wave)

            # Temporal decay
            time_diff = now - memory.timestamp
            temporal_score = np.exp(-time_diff / 86400.0)  # 1 day decay

            # Combined score
            score = (1.0 - temporal_weight) * resonance + temporal_weight * temporal_score
            scores.append((score, memory))

        scores.sort(key=lambda x: x[0], reverse=True)
        return [mem for _, mem in scores[:top_k]]

    def find_resonance(
        self,
        content: str,
        top_k: int = 5
    ) -> List[Tuple[MemoryWave, float]]:
        """
        Find memories that resonate with given content.

        Uses text similarity combined with emotional resonance.
        """
        # Simple keyword matching for now
        content_lower = content.lower()
        scores = []

        for memory in self.memories:
            # Text similarity (simple)
            text_score = 0.0
            memory_lower = memory.content.lower()
            common_words = set(content_lower.split()) & set(memory_lower.split())
            if common_words:
                text_score = len(common_words) / max(
                    len(content_lower.split()),
                    len(memory_lower.split())
                )

            scores.append((memory, text_score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def get_context(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent memories as context."""
        result = []
        for memory in self.memories[:limit]:
            emotion, intensity = memory.emotion
            result.append({
                "role": memory.role,
                "content": memory.content,
                "emotion": emotion,
                "intensity": float(intensity),
                "timestamp": memory.timestamp,
                "memory_id": memory.memory_id
            })
        return result

    def get_emotional_state(self) -> Dict[str, float]:
        """
        Get current emotional state based on recent memories.

        Returns aggregated emotional vector from recent memories.
        """
        if not self.memories:
            return {dim: 0.0 for dim in EmotionalSpace.DIMENSIONS}

        # Weight recent memories more
        now = time.time()
        weighted_sum = np.zeros(EmotionalSpace.DIM_COUNT)
        total_weight = 0.0

        for memory in self.memories[:20]:
            time_diff = now - memory.timestamp
            weight = np.exp(-time_diff / 3600.0)  # 1 hour decay
            weighted_sum += memory.wave.position * weight
            total_weight += weight

        if total_weight > 0:
            weighted_sum /= total_weight

        return {
            dim: float(weighted_sum[i])
            for i, dim in enumerate(EmotionalSpace.DIMENSIONS)
        }

    def stats(self) -> Dict[str, Any]:
        """Get memory statistics."""
        if not self.memories:
            return {
                "total_memories": 0,
                "emotions": {},
                "roles": {}
            }

        emotions: Dict[str, int] = {}
        roles: Dict[str, int] = {}

        for memory in self.memories:
            emotion, _ = memory.emotion
            emotions[emotion] = emotions.get(emotion, 0) + 1
            roles[memory.role] = roles.get(memory.role, 0) + 1

        return {
            "total_memories": len(self.memories),
            "emotions": emotions,
            "roles": roles,
            "oldest": self.memories[-1].timestamp if self.memories else None,
            "newest": self.memories[0].timestamp if self.memories else None
        }


# Singleton instance
_instance: Optional[HopeEcho] = None


def get_echo(db_path: str = "data/hope_echo.db") -> HopeEcho:
    """Get or create the global HopeEcho instance."""
    global _instance
    if _instance is None:
        _instance = HopeEcho(db_path)
    return _instance
