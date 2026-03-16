from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Subject(Base):
    __tablename__ = "subjects"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(128))
    age: Mapped[int] = mapped_column(Integer)
    gender: Mapped[str] = mapped_column(String(32))
    psqi_score: Mapped[float] = mapped_column(Float, default=0.0)
    group_type: Mapped[str] = mapped_column(String(64), default="control")
    created_time: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    experiments: Mapped[list["Experiment"]] = relationship(
        back_populates="subject", cascade="all, delete-orphan"
    )


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    subject_id: Mapped[int] = mapped_column(ForeignKey("subjects.id"), index=True)
    eeg_path: Mapped[str] = mapped_column(String(512))
    midi_path: Mapped[str] = mapped_column(String(512))
    wav_path: Mapped[str] = mapped_column(String(512))
    sws_duration: Mapped[float] = mapped_column(Float, default=0.0)
    avg_delta_power: Mapped[float] = mapped_column(Float, default=0.0)
    slow_wave_density: Mapped[float] = mapped_column(Float, default=0.0)
    created_time: Mapped[datetime] = mapped_column(
        DateTime, default=lambda: datetime.now(timezone.utc)
    )

    subject: Mapped[Subject] = relationship(back_populates="experiments")
