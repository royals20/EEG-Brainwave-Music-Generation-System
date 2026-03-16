"""
Database manager: handles session creation and CRUD operations.
"""

import os
from typing import List, Optional

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from database.models import Base, Experiment, Subject
from utils.logger import logger

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "data", "eeg_system.db")


class DatabaseManager:
    """Singleton-style database access layer."""

    def __init__(self, db_url: Optional[str] = None) -> None:
        if db_url is None:
            os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
            db_url = f"sqlite:///{DB_PATH}"
        self.engine = create_engine(db_url, echo=False)
        Base.metadata.create_all(self.engine)
        self._SessionFactory = sessionmaker(bind=self.engine)
        logger.info("Database initialised: %s", db_url)

    def _session(self) -> Session:
        return self._SessionFactory()

    # ---- Subject CRUD ----

    def add_subject(self, **kwargs) -> Subject:
        with self._session() as s:
            subj = Subject(**kwargs)
            s.add(subj)
            s.commit()
            s.refresh(subj)
            logger.info("Added subject %s", subj.subject_id)
            return subj

    def get_subject(self, subject_id: str) -> Optional[Subject]:
        with self._session() as s:
            return s.query(Subject).filter_by(subject_id=subject_id).first()

    def list_subjects(self, search: str = "") -> List[Subject]:
        with self._session() as s:
            q = s.query(Subject)
            if search:
                pattern = f"%{search}%"
                q = q.filter(
                    Subject.subject_id.like(pattern)
                    | Subject.name.like(pattern)
                    | Subject.group_type.like(pattern)
                )
            return q.order_by(Subject.created_time.desc()).all()

    def update_subject(self, subject_id: str, **kwargs) -> Optional[Subject]:
        with self._session() as s:
            subj = s.query(Subject).filter_by(subject_id=subject_id).first()
            if subj is None:
                return None
            for k, v in kwargs.items():
                if hasattr(subj, k):
                    setattr(subj, k, v)
            s.commit()
            s.refresh(subj)
            logger.info("Updated subject %s", subject_id)
            return subj

    def delete_subject(self, subject_id: str) -> bool:
        with self._session() as s:
            subj = s.query(Subject).filter_by(subject_id=subject_id).first()
            if subj is None:
                return False
            s.delete(subj)
            s.commit()
            logger.info("Deleted subject %s", subject_id)
            return True

    # ---- Experiment CRUD ----

    def add_experiment(self, **kwargs) -> Experiment:
        with self._session() as s:
            exp = Experiment(**kwargs)
            s.add(exp)
            s.commit()
            s.refresh(exp)
            logger.info("Added experiment %s", exp.experiment_id)
            return exp

    def get_experiment(self, experiment_id: str) -> Optional[Experiment]:
        with self._session() as s:
            return s.query(Experiment).filter_by(experiment_id=experiment_id).first()

    def list_experiments(self, subject_id: Optional[str] = None) -> List[Experiment]:
        with self._session() as s:
            q = s.query(Experiment)
            if subject_id:
                q = q.filter_by(subject_id=subject_id)
            return q.order_by(Experiment.created_time.desc()).all()

    def update_experiment(self, experiment_id: str, **kwargs) -> Optional[Experiment]:
        with self._session() as s:
            exp = s.query(Experiment).filter_by(experiment_id=experiment_id).first()
            if exp is None:
                return None
            for k, v in kwargs.items():
                if hasattr(exp, k):
                    setattr(exp, k, v)
            s.commit()
            s.refresh(exp)
            return exp

    def export_experiments_csv(self, path: str, subject_id: Optional[str] = None) -> str:
        """Export experiment data to CSV file."""
        import csv
        experiments = self.list_experiments(subject_id)
        headers = [
            "subject_id", "experiment_id", "created_time",
            "sws_duration", "delta_power", "slow_wave_density", "slow_wave_amplitude",
        ]
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            for e in experiments:
                writer.writerow([
                    e.subject_id, e.experiment_id,
                    e.created_time.isoformat() if e.created_time else "",
                    e.sws_duration, e.delta_power,
                    e.slow_wave_density, e.slow_wave_amplitude,
                ])
        logger.info("Exported %d experiments to %s", len(experiments), path)
        return path
