import json
import shutil
import unittest
from pathlib import Path

import sys
from pathlib import Path as _Path

sys.path.insert(0, str(_Path(__file__).resolve().parents[1] / "src"))

from ai_labeller.core.io_utils import atomic_write_json, atomic_write_text


class AtomicIoTests(unittest.TestCase):
    def setUp(self):
        self.tmp_root = Path("tests/.tmp_io")
        self.tmp_root.mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        shutil.rmtree(self.tmp_root, ignore_errors=True)

    def test_atomic_write_text(self):
        path = self.tmp_root / "a.txt"
        atomic_write_text(str(path), "hello")
        self.assertEqual(path.read_text(encoding="utf-8"), "hello")

    def test_atomic_write_json(self):
        path = self.tmp_root / "a.json"
        data = {"x": 1}
        atomic_write_json(str(path), data)
        self.assertEqual(json.loads(path.read_text(encoding="utf-8")), data)


if __name__ == "__main__":
    unittest.main()
