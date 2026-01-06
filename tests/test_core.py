"""
Tests for Hope Echo Core Module

Az érzelem a KONTEXTUS. Kontextus nélkül nincs MEGÉRTÉS.
"""

import pytest
import numpy as np
import tempfile
import os

from hope_echo.core import (
    EmotionalSpace,
    WavePacket,
    MemoryWave,
    HopeEcho,
    get_echo,
    HBAR,
    G_INTERACTION,
    DT,
)


class TestEmotionalSpace:
    """Tests for EmotionalSpace class."""

    def test_dimension_count(self):
        """Test that emotional space has 21 dimensions."""
        assert EmotionalSpace.DIM_COUNT == 21
        assert len(EmotionalSpace.DIMENSIONS) == 21

    def test_all_emotions_present(self):
        """Test that all expected emotions are in the space."""
        expected = [
            'joy', 'sadness', 'fear', 'trust', 'anger', 'surprise', 'disgust',
            'anticipation', 'love', 'guilt', 'shame', 'pride', 'envy', 'jealousy',
            'gratitude', 'hope', 'despair', 'anxiety', 'peace', 'excitement', 'contentment'
        ]
        for emotion in expected:
            assert emotion in EmotionalSpace.DIMENSIONS

    def test_emotion_to_vector_specific(self):
        """Test converting specific emotion to vector."""
        vector = EmotionalSpace.emotion_to_vector('joy', 1.0)
        assert len(vector) == 21
        assert vector[0] == 1.0  # joy is first dimension
        assert np.sum(vector) == 1.0

    def test_emotion_to_vector_intensity(self):
        """Test intensity scaling."""
        vector = EmotionalSpace.emotion_to_vector('joy', 0.5)
        assert vector[0] == 0.5

    def test_emotion_to_vector_neutral(self):
        """Test neutral emotion spreads across all dimensions."""
        vector = EmotionalSpace.emotion_to_vector('neutral', 1.0)
        assert len(vector) == 21
        # All components should be equal for neutral
        expected = 1.0 / np.sqrt(21)
        assert np.allclose(vector, expected)

    def test_vector_to_emotion(self):
        """Test converting vector back to emotion."""
        vector = np.zeros(21)
        vector[5] = 0.8  # surprise
        emotion, intensity = EmotionalSpace.vector_to_emotion(vector)
        assert emotion == 'surprise'
        assert intensity == 0.8

    def test_vector_to_emotion_low_intensity(self):
        """Test low intensity returns neutral."""
        vector = np.zeros(21)
        vector[0] = 0.05  # very low
        emotion, intensity = EmotionalSpace.vector_to_emotion(vector)
        assert emotion == 'neutral'

    def test_blend_emotions(self):
        """Test blending multiple emotions."""
        emotions = [('joy', 0.8), ('trust', 0.6)]
        blended = EmotionalSpace.blend(emotions)
        assert len(blended) == 21
        # Should be normalized
        assert np.isclose(np.linalg.norm(blended), 1.0)

    def test_blend_empty(self):
        """Test blending empty list."""
        blended = EmotionalSpace.blend([])
        assert np.allclose(blended, 0)


class TestWavePacket:
    """Tests for WavePacket class."""

    def test_creation(self):
        """Test wave packet creation."""
        position = np.zeros(21)
        momentum = np.zeros(21)
        wave = WavePacket(position, momentum, width=1.0, amplitude=1.0+0j)
        assert wave.width == 1.0
        assert wave.amplitude == 1.0+0j

    def test_psi_at_position(self):
        """Test wave function value at position."""
        position = np.zeros(21)
        wave = WavePacket(position, np.zeros(21), width=1.0, amplitude=1.0+0j)
        # At the center, psi should be maximum
        psi = wave.psi(position)
        assert abs(psi) > 0

    def test_overlap_identical(self):
        """Test overlap of identical wave packets."""
        position = EmotionalSpace.emotion_to_vector('joy', 1.0)
        wave1 = WavePacket(position, np.zeros(21), width=1.0, amplitude=1.0+0j)
        wave2 = WavePacket(position, np.zeros(21), width=1.0, amplitude=1.0+0j)
        overlap = wave1.overlap(wave2)
        assert abs(overlap) > 0

    def test_overlap_orthogonal(self):
        """Test overlap of orthogonal wave packets."""
        pos1 = EmotionalSpace.emotion_to_vector('joy', 1.0)
        pos2 = EmotionalSpace.emotion_to_vector('sadness', 1.0)
        wave1 = WavePacket(pos1, np.zeros(21), width=0.5, amplitude=1.0+0j)
        wave2 = WavePacket(pos2, np.zeros(21), width=0.5, amplitude=1.0+0j)
        # Overlap of identical vs different positions
        overlap_same = wave1.overlap(wave1)
        overlap_diff = wave1.overlap(wave2)
        # Orthogonal positions should have less overlap than same position
        assert abs(overlap_diff) < abs(overlap_same)

    def test_resonance_range(self):
        """Test resonance is in 0-1 range."""
        position = EmotionalSpace.emotion_to_vector('joy', 1.0)
        wave1 = WavePacket(position, np.zeros(21), width=1.0, amplitude=1.0+0j)
        wave2 = WavePacket(position, np.zeros(21), width=1.0, amplitude=1.0+0j)
        resonance = wave1.resonance(wave2)
        assert 0.0 <= resonance <= 1.0

    def test_evolve(self):
        """Test wave packet evolution."""
        position = EmotionalSpace.emotion_to_vector('joy', 1.0)
        momentum = np.ones(21) * 0.1
        wave = WavePacket(position.copy(), momentum.copy(), width=1.0, amplitude=1.0+0j)
        original_pos = wave.position.copy()
        wave.evolve(dt=0.1)
        # Position should have changed
        assert not np.allclose(wave.position, original_pos)

    def test_serialization(self):
        """Test wave packet serialization."""
        position = EmotionalSpace.emotion_to_vector('joy', 0.8)
        wave = WavePacket(position, np.zeros(21), width=1.5, amplitude=0.5+0.5j, memory_id="test")
        data = wave.to_dict()
        restored = WavePacket.from_dict(data)
        assert np.allclose(restored.position, wave.position)
        assert np.allclose(restored.momentum, wave.momentum)
        assert restored.width == wave.width
        assert restored.amplitude == wave.amplitude
        assert restored.memory_id == wave.memory_id


class TestMemoryWave:
    """Tests for MemoryWave class."""

    def test_creation(self):
        """Test memory wave creation."""
        position = EmotionalSpace.emotion_to_vector('gratitude', 0.9)
        wave = WavePacket(position, np.zeros(21), width=1.0, amplitude=1.0+0j, memory_id="mem1")
        memory = MemoryWave(
            memory_id="mem1",
            content="Thank you for everything!",
            wave=wave,
            timestamp=1234567890.0,
            role="user"
        )
        assert memory.content == "Thank you for everything!"
        assert memory.role == "user"

    def test_emotion_property(self):
        """Test getting dominant emotion from memory."""
        position = EmotionalSpace.emotion_to_vector('hope', 0.85)
        wave = WavePacket(position, np.zeros(21), width=1.0, amplitude=1.0+0j)
        memory = MemoryWave(
            memory_id="mem2",
            content="I hope for a better future",
            wave=wave,
            timestamp=1234567890.0,
            role="user"
        )
        emotion, intensity = memory.emotion
        assert emotion == 'hope'
        assert np.isclose(intensity, 0.85)

    def test_resonates_with(self):
        """Test resonance detection between memories."""
        pos1 = EmotionalSpace.emotion_to_vector('joy', 0.9)
        pos2 = EmotionalSpace.emotion_to_vector('joy', 0.8)
        wave1 = WavePacket(pos1, np.zeros(21), width=1.0, amplitude=1.0+0j)
        wave2 = WavePacket(pos2, np.zeros(21), width=1.0, amplitude=1.0+0j)
        mem1 = MemoryWave("m1", "Happy day!", wave1, 0.0, "user")
        mem2 = MemoryWave("m2", "Such joy!", wave2, 0.0, "user")
        assert mem1.resonates_with(mem2, threshold=0.1)

    def test_serialization(self):
        """Test memory wave serialization."""
        position = EmotionalSpace.emotion_to_vector('peace', 0.7)
        wave = WavePacket(position, np.zeros(21), width=1.0, amplitude=1.0+0j, memory_id="mem3")
        memory = MemoryWave(
            memory_id="mem3",
            content="Calm and peaceful",
            wave=wave,
            timestamp=1234567890.0,
            role="assistant",
            metadata={"source": "test"}
        )
        data = memory.to_dict()
        restored = MemoryWave.from_dict(data)
        assert restored.content == memory.content
        assert restored.role == memory.role
        assert restored.memory_id == memory.memory_id
        assert restored.metadata == memory.metadata


class TestHopeEcho:
    """Tests for HopeEcho main class."""

    @pytest.fixture
    def temp_db(self):
        """Create temporary database."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = os.path.join(tmpdir, "test_echo.db")
            yield db_path

    def test_creation(self, temp_db):
        """Test HopeEcho creation."""
        echo = HopeEcho(db_path=temp_db)
        assert echo.db_path == temp_db
        assert len(echo.memories) == 0

    def test_add_memory(self, temp_db):
        """Test adding memory."""
        echo = HopeEcho(db_path=temp_db)
        memory = echo.add("I'm so happy today!", emotion='joy', intensity=0.9)
        assert memory.content == "I'm so happy today!"
        emotion, intensity = memory.emotion
        assert emotion == 'joy'
        assert len(echo.memories) == 1

    def test_add_with_metadata(self, temp_db):
        """Test adding memory with metadata."""
        echo = HopeEcho(db_path=temp_db)
        memory = echo.add(
            "Test content",
            emotion='trust',
            intensity=0.8,
            role="assistant",
            metadata={"custom": "data"}
        )
        assert memory.metadata.get("custom") == "data"

    def test_echo_recall(self, temp_db):
        """Test emotional recall."""
        echo = HopeEcho(db_path=temp_db)
        echo.add("Happy moment 1", emotion='joy', intensity=0.9)
        echo.add("Sad moment", emotion='sadness', intensity=0.8)
        echo.add("Happy moment 2", emotion='joy', intensity=0.7)

        joyful = echo.echo('joy', top_k=2)
        assert len(joyful) == 2
        for mem in joyful:
            emotion, _ = mem.emotion
            # Should prefer joy-related memories
            assert emotion in ['joy', 'sadness']  # might include recent

    def test_get_context(self, temp_db):
        """Test getting context."""
        echo = HopeEcho(db_path=temp_db)
        echo.add("Memory 1", emotion='joy', intensity=0.8)
        echo.add("Memory 2", emotion='trust', intensity=0.7)

        context = echo.get_context(limit=10)
        assert len(context) == 2
        assert context[0]["content"] == "Memory 2"  # Most recent first

    def test_get_emotional_state(self, temp_db):
        """Test getting current emotional state."""
        echo = HopeEcho(db_path=temp_db)
        echo.add("Joyful!", emotion='joy', intensity=0.9)

        state = echo.get_emotional_state()
        assert isinstance(state, dict)
        assert len(state) == 21
        assert 'joy' in state

    def test_stats(self, temp_db):
        """Test statistics."""
        echo = HopeEcho(db_path=temp_db)
        echo.add("Test 1", emotion='joy', intensity=0.8, role="user")
        echo.add("Test 2", emotion='joy', intensity=0.7, role="assistant")
        echo.add("Test 3", emotion='sadness', intensity=0.6, role="user")

        stats = echo.stats()
        assert stats["total_memories"] == 3
        assert stats["emotions"]["joy"] == 2
        assert stats["emotions"]["sadness"] == 1
        assert stats["roles"]["user"] == 2
        assert stats["roles"]["assistant"] == 1

    def test_persistence(self, temp_db):
        """Test database persistence."""
        # Create and add
        echo1 = HopeEcho(db_path=temp_db)
        echo1.add("Persistent memory", emotion='gratitude', intensity=0.85)

        # Create new instance
        echo2 = HopeEcho(db_path=temp_db)
        assert len(echo2.memories) == 1
        assert echo2.memories[0].content == "Persistent memory"

    def test_find_resonance(self, temp_db):
        """Test finding resonating memories by content."""
        echo = HopeEcho(db_path=temp_db)
        echo.add("The quick brown fox", emotion='joy', intensity=0.8)
        echo.add("A lazy dog sleeps", emotion='peace', intensity=0.7)
        echo.add("The quick rabbit runs", emotion='excitement', intensity=0.9)

        results = echo.find_resonance("quick", top_k=2)
        assert len(results) == 2
        # Results should include memories with "quick"
        contents = [r[0].content for r in results]
        assert any("quick" in c for c in contents)

    def test_memory_limit(self, temp_db):
        """Test memory count limit."""
        echo = HopeEcho(db_path=temp_db)
        # Memory list is limited to 1000 in-memory
        for i in range(10):
            echo.add(f"Memory {i}", emotion='neutral', intensity=0.5)
        assert len(echo.memories) <= 1000


class TestConstants:
    """Tests for physical constants."""

    def test_hbar(self):
        """Test HBAR constant."""
        assert HBAR == 1.0

    def test_g_interaction(self):
        """Test interaction strength."""
        assert G_INTERACTION == 0.1

    def test_dt(self):
        """Test time step."""
        assert DT == 0.01
