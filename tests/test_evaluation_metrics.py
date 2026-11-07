import unittest

from services.evaluation.metrics import (
    compute_csat,
    compute_escalation_rate,
    compute_task_completion_rate,
    estimate_cost_per_task,
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


class RetrievalMetricsTests(unittest.TestCase):
    def test_precision_and_recall_at_k(self):
        retrieved = ["hr", "it", "finance"]
        relevant = ["it", "finance"]

        self.assertEqual(precision_at_k(retrieved, relevant, 2), 0.5)
        self.assertEqual(recall_at_k(retrieved, relevant, 2), 0.5)

    def test_retrieval_metrics_return_zero_for_empty_or_invalid_inputs(self):
        self.assertEqual(precision_at_k([], ["hr"], 5), 0.0)
        self.assertEqual(precision_at_k(["hr"], ["hr"], 0), 0.0)
        self.assertEqual(recall_at_k(["hr"], [], 5), 0.0)
        self.assertEqual(recall_at_k(["hr"], ["hr"], 0), 0.0)
        self.assertEqual(mean_reciprocal_rank([], []), 0.0)
        self.assertEqual(ndcg_at_k(["hr"], [], 5), 0.0)

    def test_mean_reciprocal_rank_averages_first_relevant_positions(self):
        score = mean_reciprocal_rank(
            [["irrelevant", "hr"], ["it"], ["none"]],
            [["hr"], ["it"], ["finance"]],
        )

        self.assertEqual(score, 0.5)

    def test_ndcg_rewards_relevant_documents_ranked_first(self):
        score = ndcg_at_k(["hr", "it", "other"], ["hr", "it"], 3)

        self.assertEqual(score, 1.0)
        self.assertLess(ndcg_at_k(["other", "hr", "it"], ["hr", "it"], 3), 1.0)


class BusinessMetricsTests(unittest.TestCase):
    def test_csat_supports_documented_feedback_shapes(self):
        result = compute_csat([
            {"is_positive": True},
            {"rating": 4},
            {"sentiment": "helpful"},
            {"score": 0.8},
            {"is_positive": False, "rating": 2},
        ])

        self.assertEqual(result, {"positive": 4, "total": 5, "percent": 80.0})
        self.assertEqual(compute_csat([]), {"positive": 0, "total": 0, "percent": 0.0})

    def test_task_completion_and_escalation_rates(self):
        messages = [
            {"response": "Your benefits begin on your first day."},
            {"answer": "Please contact HR for help with this request."},
        ]

        self.assertEqual(compute_task_completion_rate(messages), 0.5)
        self.assertEqual(compute_escalation_rate(messages), 0.5)
        self.assertEqual(compute_task_completion_rate([]), 1.0)
        self.assertEqual(compute_escalation_rate([]), 0.0)

    def test_estimated_cost_uses_recorded_token_count(self):
        messages = [{"metadata": {"total_tokens": 1500}}]

        self.assertEqual(estimate_cost_per_task(messages), 0.0015)
        self.assertEqual(estimate_cost_per_task([]), 0.0)


if __name__ == "__main__":
    unittest.main()
