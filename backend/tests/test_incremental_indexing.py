"""
Regression tests for incremental re-indexing.

create_index(previous_index_path=...) must reuse chunks + FAISS vectors for
files that haven't changed (same path/size/mtime) and only re-embed files that
are new or modified. Reuse must be refused when the embedding model changes.

Uses REAL faiss (no module-level mocks — unlike test_indexing.py) and a
deterministic fake embedding client, so no models are downloaded.
"""

import hashlib
import os
import shutil
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def setUpModule():
    from backend import database
    global original_db_path, temp_module_dir, _saved_faiss_attrs
    temp_module_dir = tempfile.mkdtemp()
    original_db_path = database.DATABASE_PATH
    database.DATABASE_PATH = os.path.join(temp_module_dir, "test_metadata.db")
    database.init_database()

    # test_indexing.py replaces faiss.IndexFlatL2/write_index/read_index with
    # module-level MagicMocks that leak into every module of the suite. These
    # tests need the REAL implementations (they round-trip actual vectors), so
    # restore them via reload for the duration of this module and put the
    # suite's previous state back afterwards.
    import importlib
    import faiss as _faiss
    _saved_faiss_attrs = {
        k: getattr(_faiss, k) for k in ("IndexFlatL2", "write_index", "read_index")
    }
    importlib.reload(_faiss)


def tearDownModule():
    from backend import database
    import faiss as _faiss
    for _k, _v in _saved_faiss_attrs.items():
        setattr(_faiss, _k, _v)
    if hasattr(database.thread_local, "connection"):
        database.thread_local.connection.close()
        del database.thread_local.connection
    database.DATABASE_PATH = original_db_path
    if os.path.exists(temp_module_dir):
        try:
            shutil.rmtree(temp_module_dir)
        except Exception as e:
            print(f"Warning: Could not cleanup module temp dir: {e}")


class FakeEmbedder:
    """Deterministic 8-dim embedder that records every text it embeds."""

    def __init__(self, model_name="fake-model"):
        self.model_name = model_name
        self.embedded_texts = []

    def embed_documents(self, batch):
        self.embedded_texts.extend(batch)
        return [self._vec(t) for t in batch]

    def embed_query(self, text):
        return self._vec(text)

    @staticmethod
    def _vec(text):
        digest = hashlib.sha256(text.encode("utf-8", "replace")).digest()
        return [b / 255.0 for b in digest[:8]]


class TestIncrementalIndexing(unittest.TestCase):
    def setUp(self):
        from backend import database
        database.init_database()
        self.temp_dir = tempfile.mkdtemp()
        self.docs_dir = os.path.join(self.temp_dir, "docs")
        os.makedirs(self.docs_dir)
        self.index_path = os.path.join(self.temp_dir, "index.faiss")
        self.ckpt_path = os.path.join(self.temp_dir, "index_checkpoint.json")

        self.file_a = os.path.join(self.docs_dir, "a.txt")
        self.file_b = os.path.join(self.docs_dir, "b.txt")
        with open(self.file_a, "w") as f:
            f.write("Alpha document about apples and orchards.")
        with open(self.file_b, "w") as f:
            f.write("Beta document about boats and harbours.")
        # Pin mtimes so fingerprint comparisons are deterministic
        _base = time.time() - 100
        os.utime(self.file_a, (_base, _base))
        os.utime(self.file_b, (_base, _base))

    def tearDown(self):
        if os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
            except Exception:
                pass

    def _index_once(self, embedder, previous=None):
        from backend import indexing
        with patch.object(indexing, "_CHECKPOINT_PATH", self.ckpt_path):
            res = indexing.create_index(
                self.docs_dir, "local",
                embedding_client=embedder,
                previous_index_path=previous,
            )
        index, docs, tags = res[0], res[1], res[2]
        meta = res[7] if len(res) > 7 else {}
        if index is not None:
            indexing.save_index(
                index, docs, tags, self.index_path,
                res[3], res[4], res[5], res[6],
                model_name=meta.get("model_name", "unknown"),
                embedding_dim=meta.get("embedding_dim", 0),
            )
        return res

    def test_unchanged_files_are_not_reembedded(self):
        first = FakeEmbedder()
        res1 = self._index_once(first)
        self.assertIsNotNone(res1[0])
        self.assertEqual(len(first.embedded_texts), len(res1[1]))  # everything embedded

        # Modify only b.txt (content + newer mtime)
        with open(self.file_b, "w") as f:
            f.write("Beta document rewritten: submarines and lighthouses.")
        newer = time.time()
        os.utime(self.file_b, (newer, newer))

        second = FakeEmbedder()
        res2 = self._index_once(second, previous=self.index_path)
        self.assertIsNotNone(res2[0])

        # Only b.txt chunks were re-embedded; a.txt was reused
        self.assertTrue(all("Beta" in t or "submarines" in t for t in second.embedded_texts),
                        f"Unexpected re-embeds: {second.embedded_texts}")
        self.assertTrue(any("submarines" in t for t in second.embedded_texts))
        self.assertFalse(any("Alpha" in t for t in second.embedded_texts),
                         "Unchanged file was re-embedded — incremental reuse failed")

        # Index stays aligned: chunk count matches vector count
        self.assertEqual(res2[0].ntotal, len(res2[1]))

        # Reused vector for a.txt must equal the original embedding
        a_positions = [c["faiss_idx"] for c in res2[1] if c["filepath"] == self.file_a]
        self.assertTrue(a_positions)
        reused_vec = res2[0].reconstruct(int(a_positions[0]))
        expected_vec = np.array(FakeEmbedder._vec(res2[1][a_positions[0]]["text"]), dtype="float32")
        np.testing.assert_allclose(reused_vec, expected_vec, rtol=1e-5)

    def test_model_change_forces_full_reembed(self):
        first = FakeEmbedder(model_name="fake-model")
        self._index_once(first)

        second = FakeEmbedder(model_name="different-model")
        res2 = self._index_once(second, previous=self.index_path)
        self.assertIsNotNone(res2[0])
        # Every chunk re-embedded because the model changed
        self.assertEqual(len(second.embedded_texts), len(res2[1]))

    def test_no_previous_index_embeds_everything(self):
        embedder = FakeEmbedder()
        res = self._index_once(embedder, previous=os.path.join(self.temp_dir, "missing.faiss"))
        self.assertIsNotNone(res[0])
        self.assertEqual(len(embedder.embedded_texts), len(res[1]))

    def test_sidecar_records_chunker_version(self):
        from backend import indexing
        self._index_once(FakeEmbedder())
        import json
        with open(os.path.splitext(self.index_path)[0] + "_meta.json") as f:
            meta = json.load(f)
        self.assertEqual(meta.get("chunker"), indexing._CHUNKER_VERSION)


if __name__ == "__main__":
    unittest.main()
