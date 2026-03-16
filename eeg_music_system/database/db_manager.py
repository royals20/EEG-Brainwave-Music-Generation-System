from pathlib import Path
from typing import Iterable, Optional

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from eeg_music_system.database.models import Base, Experiment, Subject


class DBManager:
    def __init__(self, db_path: Optional[Path] = None) -> None:
        root = Path(__file__).resolve().parents[2]
        self.db_path = db_path or (root / "eeg_music_system.db")
        self.engine = create_engine(f"sqlite:///{self.db_path}", future=True)
        self.session_factory = sessionmaker(self.engine, expire_on_commit=False)

    def initialize(self) -> None:
        Base.metadata.create_all(self.engine)

    def session(self) -> Session:
        return self.session_factory()

    def list_subjects(self, keyword: str = "") -> list[Subject]:
        with self.session() as session:
            stmt = select(Subject).order_by(Subject.created_time.desc())
            if keyword:
                stmt = stmt.where(Subject.name.contains(keyword))
            return list(session.scalars(stmt))

    def get_subject(self, subject_id: int) -> Optional[Subject]:
        with self.session() as session:
            return session.get(Subject, subject_id)

    def create_subject(
        self, name: str, age: int, gender: str, psqi_score: float, group_type: str
    ) -> Subject:
        with self.session() as session:
            subject = Subject(
                name=name,
                age=age,
                gender=gender,
                psqi_score=psqi_score,
                group_type=group_type,
            )
            session.add(subject)
            session.commit()
            session.refresh(subject)
            return subject

    def update_subject(self, subject_id: int, **kwargs) -> Optional[Subject]:
        with self.session() as session:
            subject = session.get(Subject, subject_id)
            if not subject:
                return None
            for key, value in kwargs.items():
                if hasattr(subject, key):
                    setattr(subject, key, value)
            session.commit()
            session.refresh(subject)
            return subject

    def delete_subject(self, subject_id: int) -> bool:
        with self.session() as session:
            subject = session.get(Subject, subject_id)
            if not subject:
                return False
            session.delete(subject)
            session.commit()
            return True

    def add_experiment(
        self,
        subject_id: int,
        eeg_path: str,
        midi_path: str,
        wav_path: str,
        sws_duration: float,
        avg_delta_power: float,
        slow_wave_density: float,
    ) -> Experiment:
        with self.session() as session:
            experiment = Experiment(
                subject_id=subject_id,
                eeg_path=eeg_path,
                midi_path=midi_path,
                wav_path=wav_path,
                sws_duration=sws_duration,
                avg_delta_power=avg_delta_power,
                slow_wave_density=slow_wave_density,
            )
            session.add(experiment)
            session.commit()
            session.refresh(experiment)
            return experiment

    def list_experiments(self, subject_id: Optional[int] = None) -> Iterable[Experiment]:
        with self.session() as session:
            stmt = select(Experiment).order_by(Experiment.created_time.desc())
            if subject_id:
                stmt = stmt.where(Experiment.subject_id == subject_id)
            return list(session.scalars(stmt))
