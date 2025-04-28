import pandas as pd
import re


def filter_relevant_reviews(asins_csv, input_csv, output_csv, categories_file):
    """
        This function for reducing the dataset size and contain only relevant dataset.
        It reads a large product review dataset in chunks and filters out rows
        based on two conditions:
        - If the product's ASIN is present in the provided ASIN list.
        - If the product's metadata contains any of the specified category keywords.

    """

    print("Loading ASINs and Categories...")
    # Load selected ASINs and categories
    asin_df = pd.read_csv(asins_csv)
    asins = set(asin_df["ASIN"].unique())
    category_keywords = list(set(asin_df["Category"].dropna().unique()))
    category_keywords = [kw.lower() for kw in category_keywords]
    category_set = set(category_keywords)

    found_asins = set()
    matched_categories = set()

    #  category keyword matching
    category_pattern = "|".join(re.escape(kw) for kw in category_keywords)

    print("Filtering reviews...")
    with pd.read_csv(input_csv, chunksize=100_000) as reader, open(output_csv, "w") as f_out:
        for i, chunk in enumerate(reader):
            chunk["cleaned_metadata"] = chunk["cleaned_metadata"].fillna("").astype(str).str.lower()

            # Filter by ASIN presence
            mask_asin = chunk["asin"].isin(asins)

            # Filter by category keywords in metadata (vectorized)
            mask_metadata = chunk["cleaned_metadata"].str.contains(category_pattern, regex=True)

            # Filtered data: rows where either condition is True
            filtered = chunk[mask_asin | mask_metadata]

            # Update ASIN tracker
            found_asins.update(filtered["asin"].dropna().unique())

            for kw in category_keywords:
                if chunk["cleaned_metadata"].str.contains(re.escape(kw), regex=True).any():
                    matched_categories.add(kw)

            if not filtered.empty:
                filtered.to_csv(f_out, index=False, header=(i == 0))  # write header only for first chunk

    # save missing asins and categories
    not_found_asins = asins - found_asins
    with open("asins_not_found.txt", "w") as f:
        for asin in not_found_asins:
            f.write(f"{asin}\n")

    not_found_categories = category_set - matched_categories
    with open(categories_file, "w") as f:
        for cat in not_found_categories:
            f.write(f"{cat}\n")

    print(f"Found {len(found_asins)} ASINs, missed {len(not_found_asins)}.")
    print(f"Found {len(matched_categories)} categories, missed {len(not_found_categories)}.")
