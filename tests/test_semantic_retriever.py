import unittest
from hybridtablerag.retrieval.semantic_retriever import SemanticRetriever

class TestSemanticRetriever(unittest.TestCase):

    def setUp(self):
        self.retriever = SemanticRetriever()
    
    def test_query(self):
        results = self.retriever.query("Which orders are from North region?", top_k=2)
        self.assertTrue(len(results) > 0)
        self.assertIn("table", results[0])
        self.assertIn("score", results[0])

if __name__ == "__main__":
    unittest.main()