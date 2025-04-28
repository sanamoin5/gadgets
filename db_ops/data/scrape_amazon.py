import json

import requests
from pprint import pprint

# Structure payload.
payload = {
    'source': 'amazon_search',
    'domain': 'de',
    'query': 'playstation console',
    'start_page': 1,
    'pages': 1,
    'parse': True
}

# Get response.
response = requests.request(
    'POST',
    'https://realtime.oxylabs.io/v1/queries',
    auth=('user', 'password'),
    json=payload,
)

# Print prettified response to stdout.
resp = response.json()
pprint(resp)

# Save the JSON response to a file in proper JSON format
with open('playstation_console_amazon.json', 'w') as f:
    json.dump(resp, f, indent=4)  # Use `json.dump` for valid JSON formatting

print("Response saved to 'playstation_console.json'")