"""
SQLAlchemy ORM models for the EEG Sleep Music system.
"""

from datetime import datetime

from sqlalchemy import Column, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

Base = declarative_base()


class Subject(Base):
    """Represents a study participant."""
    __tablename__ = "subjects"

    id = Column(Integer, primary_key=True, autoincrement=True)
    subject_id = Column(String(64), unique=True, nullable=False, index=True)
    name = Column(String(128), nullable=False)
    age = Column(Integer)
    gender = Column(String(16))
    psqi_score = Column(Float)
    group_type = Column(String(64))
    created_time = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Subject(subject_id={self.subject_id!r}, name={self.name!r})>"


class Experiment(Base):
    """Represents one analysis session for a subject."""
    __tablename__ = "experiments"

    id = Column(Integer, primary_key=True, autoincrement=True)
    experiment_id = Column(String(64), unique=True, nullable=False, index=True)
    subject_id = Column(String(64), nullable=False, index=True)
    eeg_path = Column(Text)
    midi_path = Column(Text)
    wav_path = Column(Text)
    sws_duration = Column(Float)
    delta_power = Column(Float)
    slow_wave_density = Column(Float)
    slow_wave_amplitude = Column(Float)
    created_time = Column(DateTime, default=datetime.utcnow)

    def __repr__(self) -> str:
        return f"<Experiment(experiment_id={self.experiment_id!r}, subject={self.subject_id!r})>"
