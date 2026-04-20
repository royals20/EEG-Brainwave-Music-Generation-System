import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from database.db_manager import DatabaseManager
from utils.config import validate_subject_id


@pytest.mark.parametrize('subject_id', ['..\\escape', 'A:B', 'bad/name', 'bad\\name'])
def test_validate_subject_id_rejects_unsafe_values(subject_id):
    with pytest.raises(ValueError):
        validate_subject_id(subject_id)


def test_validate_subject_id_accepts_trimmed_normal_id():
    assert validate_subject_id(' S001 ') == 'S001'


def test_database_manager_rejects_invalid_subject_id(tmp_path):
    db = DatabaseManager(
        db_path=str(tmp_path / 'test.db'),
        output_dir=str(tmp_path / 'output'),
    )

    assert not db.add_subject('A:B', 'Invalid Subject')
    assert db.last_error is not None
    assert '受试者 ID' in db.last_error
