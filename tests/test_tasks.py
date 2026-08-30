import unittest

from costeer.tasks import DATASET_SPECS, prepare_example


class DatasetAdapterTest(unittest.TestCase):
    def test_all_eight_datasets_are_registered(self) -> None:
        self.assertEqual(len(DATASET_SPECS), 8)

    def test_abstract_history_prompt(self) -> None:
        example = prepare_example(
            "longlamp_abstract",
            {
                "id": 1,
                "input": "Write an abstract.",
                "top_5": [{"title": "Prior title", "abstract": "Prior abstract"}],
            },
            0,
        )
        self.assertEqual(example.input_text, "Write an abstract.")
        self.assertIn("Title[1]: Prior title", example.personalized_input_text)
        self.assertIn("Abstract[1]: Prior abstract", example.personalized_input_text)
        self.assertTrue(example.personalized_input_text.endswith("Write an abstract."))

    def test_preference_prompt_matches_experiment_protocol(self) -> None:
        example = prepare_example(
            "helpsteer", {"id": 4, "input": "Explain gravity."}, 0, "concise"
        )
        self.assertEqual(
            example.personalized_input_text,
            "Explain gravity.\nYour answer should be as concise as possible",
        )

    def test_cogenesis_uses_paired_prompt(self) -> None:
        example = prepare_example(
            "cogenesis",
            {"id": "test-1", "input": "Plain", "personalized_input": "Private"},
            0,
        )
        self.assertEqual(example.input_text, "Plain")
        self.assertEqual(example.personalized_input_text, "Private")


if __name__ == "__main__":
    unittest.main()
