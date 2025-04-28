from pathlib import Path


def get_data_files(data_dir):
    """
    Auto-detect train, validation, and test CSV files in the provided directory.
    """
    data_dir = Path(data_dir)

    data_files = {}
    for split in ["train", "val", "test"]:
        matching = list(data_dir.glob(f"*_{split}.csv"))
        if matching:
            data_files[split] = str(matching[0])
        else:
            raise FileNotFoundError(f"No {split}.csv file found in {data_dir}")
    return data_files
