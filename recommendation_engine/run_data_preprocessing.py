import time

from preprocessing.cleaner import stream_clean_jsonl
from preprocessing.merger import merge_reviews_with_metadata
from preprocessing.splitter import split_dataset
from preprocessing.filter_relevant_reviews import filter_relevant_reviews

from recommendation_engine.configs.config_sbert_mps import config
from recommendation_engine.preprocessing.tokenize_and_save import tokenize_and_save_tensors


def data_preprocessing_pipeline():
    dataset_name = "electronics"
    asins_csv = config["preprocessed_dir"] / "products_consolidated.csv"
    reviews_jsonl = config["raw_data_dir"] / "Electronics.jsonl"
    metadata_jsonl = config["raw_data_dir"] / "meta_Electronics.jsonl"

    preprocessed_reviews = config["preprocessed_dir"] / f"{dataset_name}_reviews_cleaned.csv"
    preprocessed_metadata = config["preprocessed_dir"] / f"{dataset_name}_meta_cleaned.csv"
    merged_output = config["preprocessed_dir"] / "merged_electronics.csv"
    filtered_output = config["preprocessed_dir"] / "merged_filtered_reviews_asin_keywordmatch.csv"
    categories_file = config["preprocessed_dir"] / "categories_not_found.txt"
    splits_dir = config["preprocessed_dir"] / "splits"

    print("Cleaning raw data...")
    stream_clean_jsonl(str(reviews_jsonl), str(preprocessed_reviews), mode="review")
    stream_clean_jsonl(str(metadata_jsonl), str(preprocessed_metadata), mode="metadata")
    time.sleep(10)

    print("Merging datasets...")
    merge_reviews_with_metadata(str(preprocessed_reviews), str(preprocessed_metadata), str(merged_output))
    time.sleep(10)

    print("Filtering relevant reviews...")
    filter_relevant_reviews(str(asins_csv), str(merged_output), str(filtered_output), str(categories_file))
    time.sleep(10)

    print("Splitting dataset...")
    split_dataset(str(filtered_output), output_dir=str(splits_dir), output_prefix=dataset_name)
    time.sleep(10)

    splits = [
        f"{dataset_name}_train.csv",
        f"{dataset_name}_val.csv",
        f"{dataset_name}_test.csv"
    ]
    for split_file in splits:
        tokenize_and_save_tensors(
            csv_file=str(splits_dir / split_file),
            output_dir=str(config["tokenized_dir"]),
            tokenizer_name=config["tokenizer_name"],
            max_length=config["tokenizer_max_length"],
            batch_size=config["tokenizer_batch_size"]
        )


if __name__ == '__main__':
    # data_preprocessing_pipeline()
    print(f"Data preprocessing completed!! All outputs saved in: {config['base_output_dir']}")


