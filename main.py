from annotator import GPTBatchAnnotator

if __name__ == "__main__":
    annotator = GPTBatchAnnotator("images folder", "Pydantic Schema Class")
    annotator.create_batch_files()  # convert image folder into batches
    annotator.upload_batches()  # uploads the batches
    annotator.get_status()  # to check on the status of created batches
    annotator.retrieve_results()  # only call once all batches are fnisihed
    annotator.extract_labels()  # extracts the labels from the results files
