import apache_beam as beam
from apache_beam.io.parquetio import ReadFromParquet
import os
import time
import argparse
import apache_beam.metrics.metric as metrics
from apache_beam.runners.runner import PipelineResult
from datetime import datetime
from apache_beam.options.pipeline_options import PipelineOptions, StandardOptions, DirectOptions, WorkerOptions
import multiprocessing
                
def create_pipeline_options(num_workers=None):
    if num_workers is None:
        # Use number of CPU cores available
        num_workers = multiprocessing.cpu_count()
    
    options = PipelineOptions()
    
    # Set up standard options
    standard_options = options.view_as(StandardOptions)
    standard_options.streaming = False
    
    # Configure Direct Runner for parallel processing
    direct_options = options.view_as(DirectOptions)
    direct_options.direct_num_workers = num_workers
    direct_options.direct_running_mode = 'multi_processing'
    
    # Configure worker options
    worker_options = options.view_as(WorkerOptions)
    worker_options.num_workers = num_workers
    return options

class FormatRowToString(beam.DoFn):
    def __init__(self, column_name):
        self.column_name = column_name
        
    def process(self, element):
        text = element[self.column_name]
        if text.strip("\n").strip():
            yield text

def run_parallel_pipeline(languages, input_base, output_base, chunk_size, pipeline_options=None, column_name="text"):
    options = pipeline_options or beam.options.pipeline_options.PipelineOptions()
    num_cores = multiprocessing.cpu_count()
    print(f"\nStarting pipeline at {datetime.now()}")
    print(f"Number of CPU cores available: {num_cores}")
    print(f"Processing languages: {', '.join(languages)}")
    
    # Create optimized pipeline options
    options = create_pipeline_options()
    
    with beam.Pipeline('DirectRunner', options=options) as pipeline:
        # Process all languages in parallel by creating separate branches
        pcols = []
        for language in languages:
            print(f"Processing started for Language : {language}")
            pcol = (pipeline 
                | f'read_data_{language}' >> ReadFromParquet(f"{input_base}/{language}/**/**.parquet")
                | f'filter_rows_{language}' >> beam.ParDo(FormatRowToString(column_name=column_name))
                | f'write_to_text_{language}' >> beam.io.WriteToText(
                    file_path_prefix=os.path.join(output_base, language, "data"),
                    file_name_suffix="-processed.txt",
                    append_trailing_newlines=False,
                    max_records_per_shard=chunk_size,
                    skip_if_empty=True
                )
            )
            pcols.append(pcol)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--languages", type=str, default="en",  help="multiple languages can be provided as comma seperated format")
    parser.add_argument("--chunk_size", type=int, default=35_000)
    parser.add_argument("--text_column", type=str, default="text")

    args = parser.parse_args()
    
    languages = args.languages.strip().split(",")

    # Optional: Configure pipeline options
    options = beam.options.pipeline_options.PipelineOptions()
    options.view_as(beam.options.pipeline_options.StandardOptions).streaming = False

    # Run the parallel pipeline
    print("\n=== Starting Pipeline Execution ===")
    start_time = time.time()
    result = run_parallel_pipeline(
        languages=languages,
        input_base=args.input_path,
        output_base=args.output_path,
        chunk_size=args.chunk_size,
        pipeline_options=options,
        column_name=args.text_column
    )

    print("\n==== Ending Excecution ======")
    print("Time taken to complet :: ", round(time.time() - start_time, 2))
    
    # python data_processing/parquet/beam_text_writer.py --input_path=cc_100_en/statmt/cc100 --chunk_size=100000 
    # --output_path=processed_path --languages=en,da,de,es,fr,it,nl,pl,pt,sv