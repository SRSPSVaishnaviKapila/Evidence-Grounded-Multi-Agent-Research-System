"""Regression tests using real Chroma validation without persistent storage."""

import importlib.util
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import numpy as np
from chromadb.api.types import normalize_embeddings


class VectorStoreTests(unittest.TestCase):
    def setUp(self):
        self.collection = Mock()
        path = Path(__file__).resolve().parents[1] / 'app/services/vector_store_service.py'
        spec = importlib.util.spec_from_file_location('vector_store_under_test', path)
        self.service = importlib.util.module_from_spec(spec)
        with patch('chromadb.PersistentClient') as client:
            client.return_value.get_or_create_collection.return_value = self.collection
            spec.loader.exec_module(self.service)

    def test_store_accepts_numpy_float32_embeddings(self):
        chunks = [{'chunk_id': 'P1_C1', 'paper_id': 1,
                   'page_number': 1, 'text': 'Research evidence'}]
        # Cover both lists of NumPy vectors and a NumPy matrix.
        for embeddings in ([np.array([0.1, 0.2], dtype=np.float32)],
                           np.array([[0.1, 0.2]], dtype=np.float32)):
            with self.subTest(container=type(embeddings).__name__):
                self.collection.upsert.side_effect = lambda **kw: normalize_embeddings(kw['embeddings'])
                with patch.object(self.service, 'generate_embeddings', return_value=embeddings):
                    self.assertEqual(self.service.store_chunks(chunks), 1)
                args = self.collection.upsert.call_args.kwargs
                self.assertEqual(args['ids'], ['P1_C1'])
                self.assertEqual(args['documents'], ['Research evidence'])

    def test_search_accepts_numpy_float32_embedding(self):
        def query(**kwargs):
            normalize_embeddings(kwargs['query_embeddings'])
            self.assertEqual(kwargs['where'], {'paper_id': 1})
            return {'documents': [[]]}

        self.collection.query.side_effect = query
        with patch.object(self.service, 'generate_query_embedding',
                          return_value=np.array([0.1, 0.2], dtype=np.float32)):
            self.assertEqual(self.service.search_chunks('research', paper_id=1), [])

    def test_missing_embeddings_do_not_write(self):
        chunk = {'chunk_id': 'P1_C1', 'paper_id': 1,
                 'page_number': 1, 'text': 'Research evidence'}
        for embeddings in (None, [], np.empty((0, 2))):
            with patch.object(self.service, 'generate_embeddings', return_value=embeddings):
                with self.assertRaisesRegex(ValueError, 'No embeddings'):
                    self.service.store_chunks([chunk])
        self.collection.upsert.assert_not_called()


if __name__ == '__main__':
    unittest.main()
