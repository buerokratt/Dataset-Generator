# test_bulk_generation.py
import requests
import json

API_URL = "http://localhost:8000/generate-bulk"

payload = {
    "datasets": [
        {
            "agency_name": "id.ee",
            "agency_id": "1234",
            "data_path": "data/ID",
            "output_filename": "bulk_test_output"
        },
        {
            "agency_name": "Politsei- ja Piirivalveamet",
            "agency_id": "234",
            "data_path": "data/Politsei-_ja_Piirivalveamet",
            "output_filename": "bulk_test_output"
        },
    ]
}

headers = {"Content-Type": "application/json"}

response = requests.post(API_URL, data=json.dumps(payload), headers=headers, timeout=60)

print("Status code:", response.status_code)
print("Response:")
print(json.dumps(response.json(), indent=2))