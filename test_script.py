import requests

def test_prediction(file_path):
    url = "http://localhost:8000/predict/"
    
    # Check if file exists
    try:
        with open(file_path, "rb") as f:
            files = {"file": (file_path, f, "audio/wav")}
            response = requests.post(url, files=files)
            
            # Print response details for debugging
            print(f"Status Code: {response.status_code}")
            print(f"Response Text: {response.text}")
            
            if response.status_code == 200:
                return response.json()
            else:
                print(f"Error: Server returned status code {response.status_code}")
                return None
    except FileNotFoundError:
        print(f"Error: File not found: {file_path}")
        return None
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Is it running?")
        return None

if __name__ == "__main__":
    response = test_prediction("fake_examples/1.wav")
    if response:
        print("Prediction:", response)