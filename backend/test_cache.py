import unittest
import json
from unittest.mock import patch

from backend.cache import (
    normalize_question,
    question_cache_field,
    semantic_match_is_safe,
)


class CachePrimitiveTests(unittest.TestCase):
    def test_normalizes_case_unicode_and_whitespace(self):
        self.assertEqual(
            normalize_question("  Ｗhat\t IS   Paris? "),
            "what is paris?",
        )

    def test_exact_key_is_stable_but_document_scoped(self):
        self.assertEqual(
            question_cache_field("  What IS it? ", "version-a"),
            question_cache_field("what is it?", "version-a"),
        )
        self.assertNotEqual(
            question_cache_field("what is it?", "version-a"),
            question_cache_field("what is it?", "version-b"),
        )

    def test_semantic_match_accepts_close_rephrasing(self):
        self.assertTrue(
            semantic_match_is_safe(
                "What is the population of Paris?",
                "Tell me the population in Paris",
                [1.0, 0.0],
                [0.999, 0.01],
                0.98,
            )
        )

    def test_semantic_match_rejects_changed_topic_or_detail(self):
        embedding = [1.0, 0.0]
        close_embedding = [0.999, 0.01]
        self.assertFalse(
            semantic_match_is_safe(
                "What is the population of Paris?",
                "What is the area of Paris?",
                embedding,
                close_embedding,
                0.98,
            )
        )
        self.assertFalse(
            semantic_match_is_safe(
                "Was the value 10 or more?",
                "Was the value 20 or more?",
                embedding,
                close_embedding,
                0.98,
            )
        )
        self.assertFalse(
            semantic_match_is_safe(
                "What is the population of Paris?",
                "What is the population of Paris?",
                embedding,
                ["not-a-number", 0.0],
                0.98,
            )
        )


class CacheHitRouteTests(unittest.TestCase):
    def test_exact_hit_skips_embeddings_and_returns_citations(self):
        from backend import main

        class FakeRedis:
            def __init__(self):
                self.answers = {}
                self.stats = {}

            def hget(self, key, field):
                return self.answers.get((key, field))

            def hgetall(self, key):
                return self.stats

            def pipeline(self):
                return self

            def hincrby(self, key, field, increment):
                self.stats[field] = self.stats.get(field, 0) + increment

            def expire(self, *_args):
                return True

            def exec(self):
                return []

        session_id = "session-test"
        version = "document-version"
        question = "  What IS in the file? "
        field = main.question_cache_field(question, version)
        redis = FakeRedis()
        redis.answers[(main.session_answers_key(session_id), field)] = json.dumps(
            {
                "document_version": version,
                "question": "what is in the file?",
                "answer": "Cached answer",
                "sources": ["guide.pdf"],
                "citations": [{
                    "source": "guide.pdf",
                    "page": 3,
                    "excerpt": "Evidence from the source.",
                }],
                "llm_called": True,
            }
        )

        with (
            patch.object(main, "get_session", return_value={
                "chunk_count": 1,
                "document_version": version,
            }),
            patch.object(main, "get_redis", return_value=redis),
            patch.object(main, "make_embeddings", side_effect=AssertionError),
        ):
            result = main.ask_question(
                session_id,
                main.QuestionRequest(question=question),
            )

        self.assertEqual(result.cache_status, "CACHE HIT")
        self.assertEqual(result.answer, "Cached answer")
        self.assertEqual(result.citations[0].page, 3)
        self.assertEqual(result.citations[0].excerpt, "Evidence from the source.")
        self.assertEqual(result.cache_stats.total_cache_hits, 1)
        self.assertEqual(result.cache_stats.avoided_llm_calls, 1)
        self.assertEqual(result.cache_stats.avoided_vector_searches, 1)

    def test_semantic_hit_skips_vector_search(self):
        from backend import main

        class FakeRedis:
            def __init__(self, answer):
                self.answer = answer
                self.stats = {}

            def hget(self, *_args):
                return None

            def hgetall(self, key):
                return self.stats if key.endswith("cache-stats") else {
                    "entry": self.answer
                }

            def pipeline(self):
                return self

            def hincrby(self, key, field, increment):
                self.stats[field] = self.stats.get(field, 0) + increment

            def expire(self, *_args):
                return True

            def exec(self):
                return []

        class FakeEmbeddings:
            def embed_query(self, _question):
                return [0.999, 0.01]

        version = "document-version"
        redis = FakeRedis(json.dumps({
            "document_version": version,
            "question": "what is the population of paris?",
            "answer": "Cached semantic answer",
            "sources": ["guide.pdf"],
            "citations": [{
                "source": "guide.pdf",
                "page": 3,
                "excerpt": "Evidence from the source.",
            }],
            "llm_called": True,
            "embedding": [1.0, 0.0],
        }))

        with (
            patch.object(main, "get_session", return_value={
                "chunk_count": 1,
                "document_version": version,
            }),
            patch.object(main, "get_redis", return_value=redis),
            patch.object(main, "make_embeddings", return_value=FakeEmbeddings()),
            patch.object(main, "SEMANTIC_CACHE_ENABLED", True),
        ):
            result = main.ask_question(
                "session-test",
                main.QuestionRequest(question="Tell me the population in Paris"),
            )

        self.assertEqual(result.cache_status, "CACHE HIT")
        self.assertEqual(result.answer, "Cached semantic answer")
        self.assertEqual(result.cache_stats.avoided_vector_searches, 1)


if __name__ == "__main__":
    unittest.main()
