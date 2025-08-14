import requests
import json
import time

API_URL = "http://localhost:8000/generate-bulk"

payload = {
    "datasets": [
        {
            "agency_name": "elron_someuuid",
            "agency_id": "1234",
            "data_path": "data/elron_someuuid",
            "output_filename": "bulk_test_output",
            "version_id": "3"
        },
        {
            "agency_name": "sm_someuuid", 
            "agency_id": "234",
            "data_path": "data/sm_someuuid",
            "output_filename": "bulk_test_output",
            "version_id": "3"
        },
    ]
}

def test_health_first():
    """Test the health endpoint first"""
    try:
        health_response = requests.get("http://localhost:8000/health", timeout=10)
        print(f"Health check status: {health_response.status_code}")
        if health_response.status_code == 200:
            print(f"Health response: {health_response.json()}")
            return True
        else:
            print(f"Health check failed: {health_response.text}")
            return False
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_bulk_generation():
    """Test the bulk generation endpoint"""
    try:
        print("Testing bulk generation...")
        response = requests.post(API_URL, json=payload, timeout=120)
        print(f"Status code: {response.status_code}")
        print("Response:")
        print(json.dumps(response.json(), indent=2))
        
        if response.status_code == 200:
            task_id = response.json().get('task_id')
            if task_id:
                print(f"\nTask queued successfully with ID: {task_id}")
                print("You can check the status using:")
                print(f"curl http://localhost:8000/task-status/{task_id}")
                
                # Optionally check status after a few seconds
                time.sleep(5)
                status_response = requests.get(f"http://localhost:8000/task-status/{task_id}")
                if status_response.status_code == 200:
                    print(f"\nCurrent task status:")
                    print(json.dumps(status_response.json(), indent=2))
        
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error: {e}")
        print("Make sure the Dataset Generator service is running on localhost:8000")
    except requests.exceptions.Timeout as e:
        print(f"Request timeout: {e}")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("Testing Dataset Generator API...")
    
    # First test health
    if test_health_first():
        print("\n" + "="*50)
        test_bulk_generation()
    else:
        print("Service health check failed. Please check if the service is running properly.")