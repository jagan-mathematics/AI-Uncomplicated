from pathlib import Path
import pickle
from typing import Any, Optional

# class CachedDatasetLoader:
    def __init__(self, cache_dir: str = ".cache/datasets"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_cache_path(self, dataset_name: str, subset_range: tuple) -> Path:
        range_str = f"{subset_range[0]}-{subset_range[1]}"
        return self.cache_dir / f"{dataset_name.replace('/', '_')}_{range_str}.pkl"

    def load_with_cache(self, dataset_name: str, split: str = "train", range_start: int = 0,
                        range_end: Optional[int] = None, force_reload: bool = False) -> Any:
        subset_range = (range_start, range_end if range_end is not None else "end")
        cache_path = self._get_cache_path(dataset_name, subset_range)

        if not force_reload and cache_path.exists():
            print(f"Loading cached dataset from {cache_path}")
            with open(cache_path, 'rb') as f:
                return pickle.load(f)

        print(f"Loading dataset from source: {dataset_name}")
        dataset = load_dataset(dataset_name)[split]

        if range_end is not None:
            dataset = dataset.select(range(range_start, range_end))
        else:
            dataset = dataset.select(range(range_start, len(dataset)))

        print(f"Caching dataset to {cache_path}")
        with open(cache_path, 'wb') as f:
            pickle.dump(dataset, f)

        return dataset