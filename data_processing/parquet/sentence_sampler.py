import argparse
import random
import os

import apache_beam as beam
from apache_beam.io import parquetio
from apache_beam.options.pipeline_options import PipelineOptions

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

nltk.download('punkt_tab')


# Function to extract the sentence from a record.
# Change 'text' to the appropriate key if your Parquet files use a different field name.
def extract_sentence(record):
    # Expecting record is a dict with a 'text' field.
    text = record.get('content')
    sentences = sent_tokenize(text)
    return [sent.strip().strip("\n") for sent in sentences]

def filter_short_sentence(record):
    return len(word_tokenize(record.strip())) >= 5 and record.count("\n") == 0

def sample_by_percentage(element, percentage=10):
    return random.random() < (percentage / 100.0)

def run(args):
    # Define your pipeline options if needed.
    options = PipelineOptions()

    with beam.Pipeline(options=options) as p:
        sampled_sentences = (
            p
            # Read from multiple parquet files; adjust the glob pattern as needed.
            | 'ReadParquetFiles' >> parquetio.ReadFromParquet(args.input_path)
            # Extract the sentence from each record.
            | 'ExtractSentence' >> beam.FlatMap(extract_sentence)
            # Filter out records that did not have a valid sentence.
            | 'FilterShortSentences' >> beam.Filter(filter_short_sentence)
            | 'Sample' >> beam.Filter(sample_by_percentage, 20)
            # | 'FlattenSample' >> beam.FlatMap(lambda x: x)
            # # (Optional) Write the sampled sentences to a text file.
            | 'WriteToText' >> beam.io.WriteToText(file_path_prefix=os.path.join(args.output_path, "data"),
                                file_name_suffix="-sampled.txt",
                                append_trailing_newlines=True,
                                max_records_per_shard=100000,
                                skip_if_empty=True)
        )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    args = parser.parse_args()

    import time

    start_time = time.time()
    run(args)
    print(f"Completed run in {time.time() - start_time}")
