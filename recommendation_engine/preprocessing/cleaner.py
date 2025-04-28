import os
import re
import html
import json
from tqdm import tqdm


def clean_text(raw_text):
    """
    Clean raw text: remove HTML, newlines, extra quotes, ensure it ends with a period.
    """
    if isinstance(raw_text, list):
        cleaned_text = ' '.join(raw_text)
    elif isinstance(raw_text, dict):
        cleaned_text = str(raw_text)
    else:
        cleaned_text = str(raw_text)

    cleaned_text = html.unescape(cleaned_text)

    # Remove HTML tags
    cleaned_text = re.sub(r'<[^>]+>', ' ', cleaned_text)

    # Remove newlines
    cleaned_text = re.sub(r'[\n\r]+', ' ', cleaned_text)

    # Collapse multiple dots
    cleaned_text = re.sub(r'(\.\s*){2,}', '. ', cleaned_text)

    # Remove extra quotes
    cleaned_text = re.sub(r'["]+', '', cleaned_text)

    # Fix spacing around punctuation
    cleaned_text = re.sub(r'\s+([?.!,])', r'\1', cleaned_text)
    cleaned_text = re.sub(r'([?.!,])([^\s])', r'\1 \2', cleaned_text)

    # Remove double spaces
    cleaned_text = re.sub(r'\s{2,}', ' ', cleaned_text)

    cleaned_text = cleaned_text.strip()

    # Ensure ends with period
    if not cleaned_text.endswith('.'):
        cleaned_text += '.'

    return cleaned_text


def clean_review(example):
    """
    Clean review after combining title and text fields.
    """
    title = example.get('title', '')
    text = example.get('text', '')
    combined = (title + ". " + text).strip() if title else text
    example['cleaned_review'] = clean_text(combined)
    return example


def clean_metadata(example):
    """
    Clean metadata by combining important fields:
    title, description, features, categories, store, details
    """
    parts = []

    # Add main category
    if example.get('main_category'):
        parts.append(example['main_category'])

    # Add title
    if example.get('title'):
        parts.append(example['title'])

    # Add description
    desc = example.get('description', [])
    if isinstance(desc, list):
        parts.append(' '.join(desc))
    elif isinstance(desc, str):
        parts.append(desc)

    # Add features
    features = example.get('features', [])
    if isinstance(features, list):
        parts.append(' '.join(features))

    # Add categories
    categories = example.get('categories', [])
    if isinstance(categories, list):
        parts.append(' '.join(categories))

    # Add store
    if example.get('store'):
        parts.append(example['store'])

    # Add details (flatten key-value pairs)
    details = example.get('details', {})
    if isinstance(details, dict):
        detail_text = ' '.join([f"{k} {v}" for k, v in details.items() if isinstance(v, str)])
        parts.append(detail_text)

    # Combine and clean
    meta_combined = ' '.join(parts).strip()
    example['cleaned_metadata'] = clean_text(meta_combined)
    return example


def stream_clean_jsonl(input_path, output_csv, mode="review"):
    """
    Stream-process large JSONL files to Clean and Write to CSV
    """
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    with open(input_path, 'r', encoding='utf-8') as f_in, open(output_csv, "w", encoding="utf-8") as f_out:
        if mode == "review":
            header = "asin,cleaned_review,rating\n"
        elif mode == "metadata":
            header = "parent_asin,cleaned_metadata,price\n"
        else:
            raise ValueError("Mode must be 'review' or 'metadata'")
        f_out.write(header)

        for i, line in enumerate(tqdm(f_in, desc=f"Processing {mode} lines")):
            try:
                data = json.loads(line.strip())
                if mode == "review":
                    data = clean_review(data)
                    asin = data.get("asin", "").strip()
                    cleaned_review = data.get("cleaned_review", "").strip()
                    rating = data.get("rating", None)
                    # combine fields and save
                    if asin and cleaned_review and rating is not None:
                        row = f'"{asin}","{cleaned_review}",{rating}\n'
                        f_out.write(row)
                elif mode == "metadata":
                    data = clean_metadata(data)
                    parent_asin = data.get("parent_asin", "").strip() # same as asin in review ds
                    cleaned_metadata = data.get("cleaned_metadata", "").strip()
                    price = data.get("price", None)
                    # combine fields and save
                    if parent_asin and cleaned_metadata:
                        price_str = str(price) if price not in [None, "None"] else ""
                        row = f'"{parent_asin}","{cleaned_metadata}",{price_str}\n'
                        f_out.write(row)
            except Exception as e:
                print(f"Skipping line {i} due to error: {e}")

    print(f"Finished processing {mode} data → Output: {output_csv}")
