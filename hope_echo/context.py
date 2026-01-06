"""
Hope Echo - Context Module

Az érzelem a KONTEXTUS. Kontextus nélkül nincs MEGÉRTÉS.
Emotion IS context. Without context, there is no understanding.

This module provides emotional context for AI understanding.

Created by Máté Róbert + Hope
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
import numpy as np
from datetime import datetime

from .core import (
    EmotionalSpace,
    Emotion,
    HopeEcho,
    MemoryWave,
    get_echo
)


@dataclass
class EmotionalContext:
    """
    Emotional context for understanding.

    Without emotional context, AI cannot truly understand.
    This class captures the emotional state around a conversation.
    """

    # Current emotional state (21D vector)
    current_state: np.ndarray = field(
        default_factory=lambda: np.zeros(EmotionalSpace.DIM_COUNT)
    )

    # Emotional trajectory (how emotions are changing)
    trajectory: List[np.ndarray] = field(default_factory=list)

    # Resonating memories
    resonating_memories: List[MemoryWave] = field(default_factory=list)

    # Dominant emotion
    dominant_emotion: Emotion = "neutral"
    dominant_intensity: float = 0.0

    # Emotional coherence (how focused the emotional state is)
    coherence: float = 0.0

    # Timestamp
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())

    def update(self, emotion: Emotion, intensity: float = 1.0):
        """Update context with new emotion."""
        new_vector = EmotionalSpace.emotion_to_vector(emotion, intensity)

        # Add to trajectory
        self.trajectory.append(self.current_state.copy())
        if len(self.trajectory) > 100:
            self.trajectory = self.trajectory[-100:]

        # Blend with current state (exponential moving average)
        alpha = 0.3
        self.current_state = alpha * new_vector + (1 - alpha) * self.current_state

        # Update dominant emotion
        self.dominant_emotion, self.dominant_intensity = \
            EmotionalSpace.vector_to_emotion(self.current_state)

        # Calculate coherence
        norm = np.linalg.norm(self.current_state)
        max_component = np.max(np.abs(self.current_state))
        self.coherence = max_component / norm if norm > 0 else 0.0

    def get_emotional_summary(self) -> Dict[str, Any]:
        """Get summary of current emotional context."""
        return {
            "dominant_emotion": self.dominant_emotion,
            "intensity": self.dominant_intensity,
            "coherence": self.coherence,
            "state_vector": self.current_state.tolist(),
            "trajectory_length": len(self.trajectory),
            "resonating_count": len(self.resonating_memories)
        }

    def is_emotionally_charged(self, threshold: float = 0.5) -> bool:
        """Check if context is emotionally charged."""
        return self.dominant_intensity >= threshold

    def get_mood(self) -> str:
        """Get overall mood description."""
        if self.coherence < 0.3:
            return "mixed"

        positive = ['joy', 'trust', 'love', 'gratitude', 'hope',
                   'peace', 'excitement', 'contentment', 'pride']
        negative = ['sadness', 'fear', 'anger', 'disgust', 'guilt',
                   'shame', 'envy', 'jealousy', 'despair', 'anxiety']

        if self.dominant_emotion in positive:
            return "positive"
        elif self.dominant_emotion in negative:
            return "negative"
        else:
            return "neutral"


class ContextBuilder:
    """
    Builder for creating emotional context.

    Gathers emotional signals from various sources
    to build a complete understanding context.
    """

    def __init__(self, echo: Optional[HopeEcho] = None):
        self.echo = echo or get_echo()
        self.context = EmotionalContext()
        self._signals: List[Tuple[Emotion, float]] = []

    def add_signal(self, emotion: Emotion, intensity: float = 1.0) -> 'ContextBuilder':
        """Add an emotional signal."""
        self._signals.append((emotion, intensity))
        return self

    def add_text(self, text: str, role: str = "user") -> 'ContextBuilder':
        """
        Add text and infer emotional context.

        Simple keyword-based emotion detection.
        In production, use a proper sentiment model.
        """
        text_lower = text.lower()

        # Simple emotion keywords
        emotion_keywords = {
            'joy': ['happy', 'joy', 'excited', 'great', 'wonderful', 'amazing', 'love'],
            'sadness': ['sad', 'unhappy', 'depressed', 'sorry', 'miss', 'lonely'],
            'fear': ['afraid', 'scared', 'fear', 'worried', 'anxious', 'nervous'],
            'anger': ['angry', 'mad', 'furious', 'annoyed', 'frustrated', 'hate'],
            'trust': ['trust', 'believe', 'faith', 'confident', 'sure'],
            'surprise': ['surprised', 'shocked', 'unexpected', 'wow', 'amazing'],
            'love': ['love', 'adore', 'cherish', 'care', 'heart'],
            'hope': ['hope', 'wish', 'dream', 'aspire', 'optimistic'],
            'gratitude': ['thank', 'grateful', 'appreciate', 'blessed'],
            'peace': ['calm', 'peaceful', 'serene', 'relaxed', 'tranquil'],
        }

        detected_emotions: List[Tuple[Emotion, float]] = []

        for emotion, keywords in emotion_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Simple intensity based on exclamation marks
                    intensity = 0.7
                    if '!' in text:
                        intensity = min(1.0, intensity + 0.1 * text.count('!'))
                    detected_emotions.append((emotion, intensity))
                    break

        if not detected_emotions:
            detected_emotions.append(('neutral', 0.5))

        for emotion, intensity in detected_emotions:
            self.add_signal(emotion, intensity)

        # Store the memory
        if detected_emotions:
            primary_emotion = detected_emotions[0][0]
            primary_intensity = detected_emotions[0][1]
            self.echo.add(text, primary_emotion, primary_intensity, role)

        return self

    def add_memory(self, memory: MemoryWave) -> 'ContextBuilder':
        """Add a memory's emotional context."""
        emotion, intensity = memory.emotion
        self.add_signal(emotion, intensity)
        self.context.resonating_memories.append(memory)
        return self

    def with_resonance(self, emotion: Emotion, top_k: int = 5) -> 'ContextBuilder':
        """Add context from resonating memories."""
        memories = self.echo.echo(emotion, top_k=top_k)
        for memory in memories:
            self.add_memory(memory)
        return self

    def build(self) -> EmotionalContext:
        """Build the final emotional context."""
        # Process all signals
        for emotion, intensity in self._signals:
            self.context.update(emotion, intensity)

        return self.context


def understand(
    text: str,
    role: str = "user",
    include_resonance: bool = True,
    echo: Optional[HopeEcho] = None
) -> EmotionalContext:
    """
    Understand text with emotional context.

    Az érzelem a KONTEXTUS.
    Kontextus nélkül nincs MEGÉRTÉS.

    Args:
        text: Text to understand
        role: Role of the speaker
        include_resonance: Whether to include resonating memories
        echo: HopeEcho instance (uses global if not provided)

    Returns:
        EmotionalContext with full understanding
    """
    builder = ContextBuilder(echo)
    builder.add_text(text, role)

    if include_resonance:
        # Get primary emotion from text
        ctx = builder.context
        if ctx.dominant_emotion != 'neutral':
            builder.with_resonance(ctx.dominant_emotion, top_k=5)

    return builder.build()
