def sbert_collate_fn(batch, tokenizer, max_length=128):
    view1_texts = [item["view1"] for item in batch]
    view2_texts = [item["view2"] for item in batch]

    encoded_view1 = tokenizer(view1_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
    encoded_view2 = tokenizer(view2_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt")

    return {
        "view1_input_ids": encoded_view1["input_ids"],
        "view1_attention_mask": encoded_view1["attention_mask"],
        "view2_input_ids": encoded_view2["input_ids"],
        "view2_attention_mask": encoded_view2["attention_mask"],
    }