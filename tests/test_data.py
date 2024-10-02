import unittest

import numpy as np
import pandas as pd
from transformers import BertTokenizer

from annotator_diversity.datasets import (
    LabelledDataItem,
    LabelledDataset,
    UnlabelledDataset,
)
from annotator_diversity.query_annotations import randomly_select_annotations
from annotator_diversity.query_datapoints import randomly_select_datapoints


class TestDataItem(unittest.TestCase):
    def setUp(self):
        # Initialize tokenizers and datasets
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.max_length = 128

        # Sample data for testing
        annotation_data = {
            "dataID": [1, 2, 2, 3],
            "annotator": ["A", "A", "B", "C"],
            "annotation": [1, 2, 3, 4],
            "text": ["Sample text"] * 4,
            "annotationID": [1, 2, 3, 4],
        }
        self.annotation_df = pd.DataFrame(annotation_data)

    def test_initialization(self):
        # create a labelled data item
        item_binary_classification = LabelledDataItem(
            index=1, text="Sample text", max_annotations=3, num_labels=2
        )
        self.assertEqual(item_binary_classification.index, 1)
        self.assertEqual(item_binary_classification.text, "Sample text")
        self.assertEqual(item_binary_classification.max_annotations, 3)
        self.assertEqual(item_binary_classification.current_annotations, 0)
        self.assertEqual(item_binary_classification.soft_label, None)
        self.assertEqual(len(item_binary_classification.annotators), 0)
        self.assertEqual(len(item_binary_classification.raw_labels), 0)
        self.assertEqual(len(item_binary_classification.annotation_ids), 0)

    def test_add_annotation(self):
        # create a labelled data item and add an annotation
        item_binary_classification = LabelledDataItem(
            index=1, text="Sample text", max_annotations=3, num_labels=2
        )
        item_binary_classification.add_annotation(
            annotation=1, annotator="Annotator1", annotation_id=1
        )
        self.assertEqual(item_binary_classification.current_annotations, 1)
        self.assertIn("Annotator1", item_binary_classification.annotators)
        # add another annotation
        item_binary_classification.add_annotation(0, "Annotator2", annotation_id=2)
        self.assertEqual(item_binary_classification.current_annotations, 2)
        self.assertIn("Annotator2", item_binary_classification.annotators)
        # validate that the soft label is calculated correctly
        expected_soft_label = 0.5
        np.testing.assert_array_almost_equal(
            item_binary_classification.soft_label, expected_soft_label, decimal=2
        )

        # create a labelled data item with multi-class classification and add an annotation
        item_multi_classification = LabelledDataItem(
            index=1, text="Sample text", max_annotations=3, num_labels=3
        )
        item_multi_classification.add_annotation(
            annotation=1, annotator="Annotator1", annotation_id=1
        )
        self.assertEqual(item_multi_classification.current_annotations, 1)
        item_multi_classification.add_annotation(
            annotation=2, annotator="Annotator2", annotation_id=2
        )
        item_multi_classification.add_annotation(
            annotation=0, annotator="Annotator3", annotation_id=3
        )
        self.assertIn("Annotator1", item_multi_classification.annotators)
        expected_soft_label = [0.33, 0.33, 0.33]
        np.testing.assert_array_almost_equal(
            item_multi_classification.soft_label, expected_soft_label, decimal=2
        )

    def test_max_annotations_limit(self):
        item_multi_classification = LabelledDataItem(
            index=1, text="Sample text", max_annotations=3, num_labels=3
        )
        item_multi_classification.add_annotation(
            annotation=1, annotator="Annotator1", annotation_id=1
        )
        item_multi_classification.add_annotation(
            annotation=2, annotator="Annotator2", annotation_id=2
        )
        item_multi_classification.add_annotation(
            annotation=0, annotator="Annotator3", annotation_id=3
        )
        # Trying to add more annotations than allowed
        with self.assertRaises(Exception):
            item_multi_classification.add_annotation(
                annotation=2, annotator="a4", annotation_id=4
            )

    def test_pop_annotation(self):
        unagg_df = pd.DataFrame(
            {
                "ID": [1, 1, 1],
                "annotator": ["Annotator1", "Annotator2", "Annotator3"],
                "annotation": [1, 2, 3],
                "text": ["Sample text"] * 3,
                "annot_id": [1, 2, 3],
            }
        )
        dataset = UnlabelledDataset(
            unaggregated_df=unagg_df,
            text_column="text",
            annotator_column="annotator",
            annotation_column="annotation",
            data_id_column="ID",
            annotation_id_column="annot_id",
        )
        sample_item = dataset.items[0]
        # Test removing an annotation and annotator
        dataset.pop_annotation(
            1, 1
        )  # Remove the second annotation and annotator from the first item
        # Check if the annotation and annotator are removed
        self.assertEqual(sample_item.annotations, [1, 3])
        self.assertEqual(sample_item.annotators, ["Annotator1", "Annotator3"])

        # Test removing the remaining annotations and annotators
        dataset.pop_annotation(1, 0)
        dataset.pop_annotation(1, 0)

        # Check if the item is removed from the dataset
        self.assertEqual(len(dataset.items), 0)

    def test_total_annotations(self):
        # create a labelled dataset
        labelled_dataset = LabelledDataset(
            max_length=self.max_length, tokenizer=self.tokenizer
        )
        # create a labelled data item
        item_binary_classification = LabelledDataItem(
            index=1, text="Sample text", max_annotations=3, num_labels=2
        )
        item_multi_classification = LabelledDataItem(
            index=1, text="Sample text", max_annotations=3, num_labels=3
        )
        # add annotations
        item_binary_classification.add_annotation(
            annotation=1, annotator="Annotator1", annotation_id=1
        )
        item_binary_classification.add_annotation(
            annotation=0, annotator="Annotator2", annotation_id=2
        )
        item_multi_classification.add_annotation(
            annotation=1, annotator="Annotator1", annotation_id=1
        )
        item_multi_classification.add_annotation(
            annotation=2, annotator="Annotator2", annotation_id=2
        )
        # add items to the dataset
        labelled_dataset.add_item(item_binary_classification)
        labelled_dataset.add_item(item_multi_classification)
        # validate the total number of annotations
        total = labelled_dataset.total_annotations()
        self.assertEqual(total, 4)

        # create an unlabelled dataset
        unlabelled_dataset = UnlabelledDataset(
            self.annotation_df,
            text_column="text",
            data_id_column="dataID",
            annotation_id_column="annotationID",
            annotator_column="annotator",
            annotation_column="annotation",
        )
        # validate the total number of annotations
        total = unlabelled_dataset.total_annotations_count()
        self.assertEqual(total, 4)

    def test_simulate_active_learning(self):
        labelled_dataset = LabelledDataset(
            max_length=self.max_length, tokenizer=self.tokenizer
        )
        annotation_data = {
            "ID": [1, 2, 2, 3],
            "annotator": ["A", "A", "B", "C"],
            "annotation": [1, 2, 3, 4],
            "text": ["Sample text"] * 4,
            "annot_id": [1, 2, 3, 4],
        }
        annotation_df = pd.DataFrame(annotation_data)

        # Create unlabelled dataset
        unlabelled_dataset = UnlabelledDataset(
            unaggregated_df=annotation_df,
            text_column="text",
            data_id_column="ID",
            annotation_id_column="annot_id",
            annotator_column="annotator",
            annotation_column="annotation",
        )
        initial_total_annotations = unlabelled_dataset.total_annotations_count()

        while unlabelled_dataset.items:
            # step 1: randomly select datapoints
            # Randomly select items (e.g., one item per iteration)
            selected_items = randomly_select_datapoints(unlabelled_dataset, 2)

            # step 2: randomly select annotations
            selected_annotations = randomly_select_annotations(selected_items)
            # update labelled and unlabelled dataset: add a new LabelledItem to the LabelledDataset if the index of that item is not yet in it and add
            # the first annotations and annotators
            # if the index is already in the LabelledDataset: update the annotation
            for i, item in enumerate(selected_items):
                annotation = selected_annotations["annotations"][i]
                annotator = selected_annotations["annotators"][i]
                annotation_id = selected_annotations["annotation_ids"][i]

                # Check if the item already exists in the labelled dataset
                if item.data_id in labelled_dataset.all_item_indices():
                    existing_item = labelled_dataset.get_item_by_index(
                        index=item.data_id
                    )
                else:
                    # Create a new item if it doesn't exist in the labelled dataset
                    existing_item = LabelledDataItem(
                        index=item.data_id,
                        text=item.text,
                        max_annotations=item.max_annotations,
                        num_labels=4,
                    )
                    labelled_dataset.add_item(existing_item)

                # Add the annotation to the item
                existing_item.add_annotation(
                    annotation=annotation,
                    annotator=annotator,
                    annotation_id=annotation_id,
                )

                # Remove the annotation from the unlabelled dataset
                unlabelled_dataset.pop_annotation(
                    item.data_id, selected_annotations["annotation_indices"][i]
                )

        # Validation checks
        assert len(unlabelled_dataset.items) == 0, "Unlabelled dataset is not empty."
        total_annotations_labelled = labelled_dataset.total_annotations()
        assert (
            total_annotations_labelled == initial_total_annotations
        ), "Not all annotations were transferred."


if __name__ == "__main__":
    unittest.main()
