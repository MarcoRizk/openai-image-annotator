from schema import Output
from annotator import GPTBatchAnnotator

if __name__ == "__main__":
    annotator = GPTBatchAnnotator("images", Output, batch_base_name="test-batch-2")
    annotator.create_batch_files()  # convert image folder into batches
    annotator.upload_batches()  # uploads the batches
    annotator.wait_for_results_and_retrieve() # waits until batches are finished processing and saves the results
    annotator.extract_labels()  # extracts the labels from the results files
