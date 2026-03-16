import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from eeg_music_system.database.db_manager import DBManager
from eeg_music_system.services.experiment_manager import ExperimentManager


class DBAndServiceTests(unittest.TestCase):
    def test_subject_crud(self):
        with tempfile.TemporaryDirectory() as tmp:
            db = DBManager(Path(tmp) / "test.db")
            db.initialize()
            s = db.create_subject("Alice", 28, "F", 5.5, "test")
            self.assertIsNotNone(s.id)

            updated = db.update_subject(s.id, name="Alice2")
            self.assertEqual(updated.name, "Alice2")
            self.assertTrue(db.delete_subject(s.id))

    def test_csv_import_and_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            db = DBManager(tmp_path / "test.db")
            manager = ExperimentManager(db)
            manager.file_manager.base_dir = tmp_path / "subjects"

            subject = db.create_subject("Bob", 35, "M", 7.0, "patient")

            data = pd.DataFrame({"eeg": np.sin(np.linspace(0, 20, 1000))})
            csv_path = tmp_path / "sample.csv"
            data.to_csv(csv_path, index=False)

            result = manager.run_analysis_and_music(subject.id, str(csv_path))
            self.assertIn("experiment_id", result)
            self.assertTrue((tmp_path / "subjects" / str(subject.id) / "brain_music.mid").exists())
            self.assertTrue((tmp_path / "subjects" / str(subject.id) / "brain_music.wav").exists())


if __name__ == "__main__":
    unittest.main()
