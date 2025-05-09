import os


def parquet_to_jsonl(dataset, work_dir, src_dir, tgt_dir, ntasks=64):
    from datatrove.executor import LocalPipelineExecutor
    from datatrove.pipeline.readers import ParquetReader
    from datatrove.pipeline.writers import JsonlWriter
    dataset_name = dataset.split("/", 1)[0]

    pipeline_exec = LocalPipelineExecutor(
        pipeline=[
            ParquetReader(
                str(src_dir),
                file_progress=True,
                doc_progress=True,
                text_key="content",
                glob_pattern="**/*.parquet",
            ),
            JsonlWriter(
                str(tgt_dir),
                output_filename=dataset_name + ".chunk.${rank}.jsonl",
                compression=None,
            ),
        ],
        tasks=ntasks,
        logging_dir=os.path.join(work_dir, "datatrove"),
    )
    pipeline_exec.run()