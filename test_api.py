import requests
import os
import argparse

def test_api(image_paths, server_url="http://localhost:5000", output_format="pdf"):
    """Test the brain tumor analysis API with provided images."""
    
    # Prepare the files for upload
    files = []
    for image_path in image_paths:
        if not os.path.exists(image_path):
            print(f"Error: Image file {image_path} not found")
            continue
        
        files.append(('images', (os.path.basename(image_path), open(image_path, 'rb'), 'image/png')))
    
    if not files:
        print("No valid image files found")
        return
    
    try:
        print(f"Uploading {len(files)} image(s) to {server_url}...")
        
        if output_format == "pdf":
            # Send request to get PDF response
            response = requests.post(f"{server_url}/analyze", files=files)
            
            if response.status_code == 200:
                # Save the PDF file
                output_filename = f"analysis_result_{int(time.time())}.pdf"
                with open(output_filename, 'wb') as f:
                    f.write(response.content)
                print(f"Success! PDF report saved as: {output_filename}")
            else:
                print(f"Error: {response.status_code} - {response.text}")
        
        else:  # JSON format
            # Send request to get JSON response
            response = requests.post(f"{server_url}/analyze-json", files=files)
            
            if response.status_code == 200:
                result = response.json()
                print("Success! Analysis results:")
                print(f"Total images analyzed: {result['summary']['total_images']}")
                print(f"Total detections: {result['summary']['total_detections']}")
                print("\nDetection Results:")
                for i, image in enumerate(result['detection_results']['images']):
                    print(f"\nImage {i+1}: {image['filename']}")
                    if image['detections']:
                        for j, detection in enumerate(image['detections']):
                            print(f"  Detection {j+1}: {detection['class']} (confidence: {detection['confidence']:.2f})")
                    else:
                        print("  No tumors detected")
                
                print(f"\nMedical Report:")
                print(result['medical_report'])
            else:
                print(f"Error: {response.status_code} - {response.text}")
    
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # Close all opened files
        for _, file_tuple in files:
            if len(file_tuple) > 1 and hasattr(file_tuple[1], 'close'):
                file_tuple[1].close()

def check_server_health(server_url="http://localhost:5000"):
    """Check if the server is running and healthy."""
    try:
        response = requests.get(f"{server_url}/health")
        if response.status_code == 200:
            health_data = response.json()
            print("Server is healthy!")
            print(f"YOLO model loaded: {health_data['models_loaded']['yolo']}")
            print(f"LLM model loaded: {health_data['models_loaded']['llm']}")
            return True
        else:
            print(f"Server health check failed: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"Cannot connect to server: {e}")
        return False

if __name__ == "__main__":
    import time
    
    parser = argparse.ArgumentParser(description="Test the brain tumor analysis API")
    parser.add_argument("--images", nargs="+", help="Paths to image files", default=["./test/Figure-A-Axial-T1-MRI-with-contrast-shows-no-evidence-of-a-brain-tumor.png"])
    parser.add_argument("--server", default="http://localhost:5000", help="Server URL")
    parser.add_argument("--format", choices=["pdf", "json"], default="pdf", help="Output format")
    parser.add_argument("--health-check", action="store_true", help="Only check server health")
    
    args = parser.parse_args()
    
    if args.health_check:
        check_server_health(args.server)
    else:
        # First check if server is healthy
        if check_server_health(args.server):
            print("\nTesting image analysis...")
            test_api(args.images, args.server, args.format)
        else:
            print("Server is not available. Please start the server first.")
