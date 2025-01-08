import os
import json
import pandas as pd

# Folder path containing JSON files
folder_path = "scraped_data"

# List to store extracted data
data = []

# Iterate through all JSON files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".json"):  # Ensure it's a JSON file
        file_path = os.path.join(folder_path, file_name)

        # Open and load the JSON file
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

            # Extract relevant information
            for result in json_data.get("results", []):
                content = result.get("content", {})
                query = content.get("query", "Unknown Category")  # Extract category from query
                organic_results = content.get("results", {}).get("organic", [])

                for item in organic_results:
                    # Extract fields for each item and include filename and category
                    data.append({
                        "Filename": file_name,  # Track which file the data came from
                        "Category": query,  # Extract category (query term)
                        "Title": item.get("title", ""),
                        "Price": item.get("price", 0),
                        "Rating": item.get("rating", 0),
                        "Currency": item.get("currency", ""),
                        "Image URL": item.get("url_image", ""),
                        "ASIN": item.get("asin", ""),
                        "URL": f"https://www.amazon.de{item.get('url', '')}",
                        "Sales Volume": item.get("sales_volume", ""),
                        "Amazon's Choice": item.get("is_amazons_choice", False),
                        "Best Seller": item.get("best_seller", False),
                        "Shipping Information": item.get("shipping_information", ""),
                        "Reviews Count": item.get("reviews_count", 0)
                    })

# Convert data into a DataFrame
df = pd.DataFrame(data)

# Save the dataset as a CSV file
output_file = "products_consolidated.csv"
df.to_csv(output_file, index=False, encoding="utf-8")

# Display the first few rows of the DataFrame
print(df.head())
