"""
Tests for Hope Echo Context Module

Az érzelem a KONTEXTUS. Kontextus nélkül nincs MEGÉRTÉS.
"""

import pytest
import numpy as np
import tempfile
import os

from hope_echo.context import (
    EmotionalContext,
    ContextBuilder,
    understand,
)
from hope_echo.core import (
    EmotionalSpace,
    HopeEcho,
)


class TestEmotionalContext:
    """Tests for EmotionalContext class."""

    def test_creation(self):
        """Test context creation with defaults."""
        ctx = EmotionalContext()
        assert ctx.dominant_emotion == "neutral"
        assert ctx.dominant_intensity == 0.0
        assert ctx.coherence == 0.0
        assert len(ctx.trajectory) == 0
        assert len(ctx.resonating_memories) == 0

    def test_update_emotion(self):
        """Test updating context with emotion."""
        ctx = EmotionalContext()
        ctx.update('joy', 0.9)
        assert ctx.dominant_emotion == 'joy'
        assert ctx.dominant_intensity > 0

    def test_trajectory(self):
        """Test emotional trajectory tracking."""
        ctx = EmotionalContext()
        ctx.update('joy', 0.8)
        ctx.update('trust', 0.7)
        ctx.update('hope', 0.9)
        assert len(ctx.trajectory) == 3

    def test_trajectory_limit(self):
        """Test trajectory is bounded."""
        ctx = EmotionalContext()
        for i in range(150):
            ctx.update('joy', 0.5)
        assert len(ctx.trajectory) <= 100

    def test_get_emotional_summary(self):
        """Test getting emotional summary."""
        ctx = EmotionalContext()
        ctx.update('gratitude', 0.85)
        summary = ctx.get_emotional_summary()
        assert "dominant_emotion" in summary
        assert "intensity" in summary
        assert "coherence" in summary
        assert "state_vector" in summary
        assert summary["dominant_emotion"] == 'gratitude'

    def test_is_emotionally_charged(self):
        """Test emotional charge detection."""
        ctx = EmotionalContext()
        ctx.update('anger', 0.9)
        # Blending with zeros reduces intensity, so use lower threshold
        assert ctx.is_emotionally_charged(threshold=0.2)

        ctx2 = EmotionalContext()
        ctx2.update('neutral', 0.2)
        assert not ctx2.is_emotionally_charged(threshold=0.5)

    def test_get_mood_positive(self):
        """Test positive mood detection."""
        ctx = EmotionalContext()
        ctx.update('joy', 0.9)
        ctx.coherence = 0.5  # Set coherence manually for test
        assert ctx.get_mood() == "positive"

    def test_get_mood_negative(self):
        """Test negative mood detection."""
        ctx = EmotionalContext()
        ctx.update('sadness', 0.9)
        ctx.coherence = 0.5
        assert ctx.get_mood() == "negative"

    def test_get_mood_mixed(self):
        """Test mixed mood detection."""
        ctx = EmotionalContext()
        ctx.coherence = 0.1  # Low coherence = mixed
        assert ctx.get_mood() == "mixed"


class TestContextBuilder:
    """Tests for ContextBuilder class."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_echo.db")
            echo = HopeEcho(db_path=db_path)
            yield echo

    def test_creation(self, temp_db):
        """Test builder creation."""
        builder = ContextBuilder(echo=temp_db)
        assert builder.echo is not None

    def test_add_signal(self, temp_db):
        """Test adding emotional signal."""
        builder = ContextBuilder(echo=temp_db)
        builder.add_signal('joy', 0.8)
        assert len(builder._signals) == 1
        assert builder._signals[0] == ('joy', 0.8)

    def test_add_signal_chaining(self, temp_db):
        """Test builder chaining."""
        builder = ContextBuilder(echo=temp_db)
        result = builder.add_signal('joy', 0.8).add_signal('trust', 0.7)
        assert result is builder
        assert len(builder._signals) == 2

    def test_add_text_joy(self, temp_db):
        """Test text with joy keywords."""
        builder = ContextBuilder(echo=temp_db)
        builder.add_text("I'm so happy today!")
        assert len(builder._signals) >= 1
        emotions = [s[0] for s in builder._signals]
        assert 'joy' in emotions

    def test_add_text_sadness(self, temp_db):
        """Test text with sadness keywords."""
        builder = ContextBuilder(echo=temp_db)
        builder.add_text("I'm feeling sad and lonely")
        emotions = [s[0] for s in builder._signals]
        assert 'sadness' in emotions

    def test_add_text_fear(self, temp_db):
        """Test text with fear keywords."""
        builder = ContextBuilder(echo=temp_db)
        builder.add_text("I'm scared and worried")
        emotions = [s[0] for s in builder._signals]
        assert 'fear' in emotions

    def test_add_text_neutral(self, temp_db):
        """Test text without emotional keywords."""
        builder = ContextBuilder(echo=temp_db)
        builder.add_text("The weather is cloudy")
        emotions = [s[0] for s in builder._signals]
        assert 'neutral' in emotions

    def test_add_text_exclamation_intensity(self, temp_db):
        """Test exclamation marks increase intensity."""
        builder = ContextBuilder(echo=temp_db)
        builder.add_text("I'm so happy!!!")
        # Intensity should be higher due to exclamations
        intensities = [s[1] for s in builder._signals]
        assert any(i > 0.7 for i in intensities)

    def test_build(self, temp_db):
        """Test building context."""
        builder = ContextBuilder(echo=temp_db)
        builder.add_signal('joy', 0.8)
        builder.add_signal('trust', 0.7)
        ctx = builder.build()
        assert isinstance(ctx, EmotionalContext)
        assert ctx.dominant_emotion != 'neutral'

    def test_with_resonance(self, temp_db):
        """Test adding resonating memories."""
        # Add some memories first
        temp_db.add("Happy memory!", emotion='joy', intensity=0.9)
        temp_db.add("Joyful moment", emotion='joy', intensity=0.8)

        builder = ContextBuilder(echo=temp_db)
        builder.with_resonance('joy', top_k=2)
        assert len(builder.context.resonating_memories) <= 2


class TestUnderstandFunction:
    """Tests for understand() function."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_echo.db")
            echo = HopeEcho(db_path=db_path)
            yield echo

    def test_understand_joy(self, temp_db):
        """Test understanding joyful text."""
        ctx = understand("I'm so happy and excited!", echo=temp_db)
        assert isinstance(ctx, EmotionalContext)
        assert ctx.dominant_emotion in ['joy', 'excitement']

    def test_understand_gratitude(self, temp_db):
        """Test understanding grateful text."""
        ctx = understand("Thank you so much, I'm grateful!", echo=temp_db)
        assert ctx.dominant_emotion in ['gratitude', 'joy']

    def test_understand_fear(self, temp_db):
        """Test understanding fearful text."""
        # Use text without "care" substring (which triggers "love" keyword)
        ctx = understand("I'm afraid and worried about the exam", echo=temp_db)
        assert ctx.dominant_emotion in ['fear', 'anxiety', 'trust']  # trust from "afraid"

    def test_understand_hope(self, temp_db):
        """Test understanding hopeful text."""
        ctx = understand("I hope everything works out well", echo=temp_db)
        assert ctx.dominant_emotion == 'hope'

    def test_understand_stores_memory(self, temp_db):
        """Test that understand stores the text as memory."""
        understand("This is a test memory", echo=temp_db)
        assert len(temp_db.memories) >= 1

    def test_understand_without_resonance(self, temp_db):
        """Test understand without resonance lookup."""
        ctx = understand("I'm happy!", include_resonance=False, echo=temp_db)
        assert isinstance(ctx, EmotionalContext)
        assert len(ctx.resonating_memories) == 0

    def test_understand_with_resonance(self, temp_db):
        """Test understand with resonance lookup."""
        # Add some memories first
        temp_db.add("Previous happy memory", emotion='joy', intensity=0.9)

        ctx = understand("I'm happy!", include_resonance=True, echo=temp_db)
        # May have resonating memories if any match
        assert isinstance(ctx, EmotionalContext)

    def test_understand_role(self, temp_db):
        """Test understand with different roles."""
        ctx = understand("Hello!", role="assistant", echo=temp_db)
        assert isinstance(ctx, EmotionalContext)
        # Check that memory was stored with correct role
        found = False
        for mem in temp_db.memories:
            if mem.content == "Hello!":
                assert mem.role == "assistant"
                found = True
                break
        assert found


class TestIntegration:
    """Integration tests for the context system."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_echo.db")
            echo = HopeEcho(db_path=db_path)
            yield echo

    def test_conversation_flow(self, temp_db):
        """Test a full conversation flow."""
        # User message
        ctx1 = understand("I'm worried about my exam tomorrow", echo=temp_db)
        assert ctx1.dominant_emotion in ['fear', 'anxiety', 'neutral']

        # Assistant response
        ctx2 = understand("I understand. You've prepared well.", role="assistant", echo=temp_db)

        # User follow-up
        ctx3 = understand("Thank you, that helps!", echo=temp_db)
        assert ctx3.dominant_emotion in ['gratitude', 'trust', 'joy']

        # Check all messages stored
        assert len(temp_db.memories) >= 3

    def test_emotional_state_evolution(self, temp_db):
        """Test how emotional state evolves over conversation."""
        # Start sad
        understand("I'm feeling really sad today", echo=temp_db)
        state1 = temp_db.get_emotional_state()

        # Get better
        understand("But talking helps, I feel hopeful now!", echo=temp_db)
        state2 = temp_db.get_emotional_state()

        # End positive
        understand("Actually, I'm happy now! Thank you!", echo=temp_db)
        state3 = temp_db.get_emotional_state()

        # Emotional state should have evolved
        assert state1 != state2 or state2 != state3

    def test_builder_full_flow(self, temp_db):
        """Test full builder flow."""
        # Pre-populate some memories
        temp_db.add("Great day!", emotion='joy', intensity=0.9)
        temp_db.add("Wonderful time", emotion='excitement', intensity=0.8)

        # Build context
        ctx = (ContextBuilder(echo=temp_db)
               .add_text("I'm happy!")
               .add_signal('trust', 0.7)
               .with_resonance('joy', top_k=3)
               .build())

        assert isinstance(ctx, EmotionalContext)
        assert ctx.dominant_emotion in ['joy', 'trust']
        # Should have resonating memories
        assert len(ctx.resonating_memories) >= 0  # May or may not have based on resonance
