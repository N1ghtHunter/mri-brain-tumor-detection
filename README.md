# Brain Tumor Analysis API

A lightweight Flask-based backend service that performs brain tumor detection on MRI images and generates medical reports in PDF format.

## Features

- **Image Upload**: Accept multiple brain MRI images (PNG, JPG, JPEG)
- **Tumor Detection**: Uses YOLO model for brain tumor detection (Glioma, Meningioma, No Tumor, Pituitary)
- **Medical Report Generation**: AI-generated medical reports using Microsoft Phi-2 model
- **PDF Report Output**: Comprehensive PDF reports with detection results and medical analysis
- **JSON API**: Alternative endpoint for programmatic access to results
- **Health Monitoring**: Health check endpoint for monitoring service status

## API Endpoints

### 1. Health Check

```
GET /health
```

Returns server status and model loading information.

### 2. Analyze Images (PDF Response)

```
POST /analyze
Content-Type: multipart/form-data
```

**Parameters:**

- `images`: Multiple image files (form-data)

**Response:** PDF file download

### 3. Analyze Images (JSON Response)

```
POST /analyze-json
Content-Type: multipart/form-data
```

**Parameters:**

- `images`: Multiple image files (form-data)

**Response:** JSON with detection results and medical report

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Ensure Models are Available

Make sure `yolov8_model.pt` is in the project directory.

### 3. Start the Server

```bash
python app.py
```

The server will start on `http://localhost:5000`

### 4. Test the API

```bash
# Test with PDF output
python test_api.py --images path/to/image.png --format pdf

# Test with JSON output
python test_api.py --images path/to/image1.png path/to/image2.png --format json

# Health check only
python test_api.py --health-check
```

## Usage Examples

### Using cURL

#### Health Check

```bash
curl http://localhost:5000/health
```

#### Analyze Images (PDF)

```bash
curl -X POST -F "images=@image1.png" -F "images=@image2.png" \
     http://localhost:5000/analyze \
     --output report.pdf
```

#### Analyze Images (JSON)

```bash
curl -X POST -F "images=@image.png" \
     http://localhost:5000/analyze-json \
     -H "Content-Type: multipart/form-data"
```

### Using Python requests

```python
import requests

# Upload images for analysis
files = [
    ('images', ('brain_scan1.png', open('brain_scan1.png', 'rb'), 'image/png')),
    ('images', ('brain_scan2.png', open('brain_scan2.png', 'rb'), 'image/png'))
]

# Get PDF report
response = requests.post('http://localhost:5000/analyze', files=files)
with open('report.pdf', 'wb') as f:
    f.write(response.content)

# Get JSON response
response = requests.post('http://localhost:5000/analyze-json', files=files)
results = response.json()
```

## Configuration

### Environment Variables

- `FLASK_ENV`: Set to `development` for debug mode
- `CUDA_VISIBLE_DEVICES`: Control GPU usage

### Model Configuration

- Default YOLO model: `yolov8_model.pt`
- Default LLM model: `microsoft/phi-2`
- Supported tumor classes: Glioma, Meningioma, No Tumor, Pituitary

## Performance Notes

- **Model Loading**: Models are loaded once at startup for optimal performance
- **Memory Usage**: LLM model requires significant GPU/CPU memory
- **File Size Limits**: Maximum upload size is 16MB per request
- **Concurrent Requests**: Single-threaded Flask server (use gunicorn for production)

## Production Deployment

For production use, consider:

1. **Use Gunicorn**:

```bash
gunicorn -w 1 -b 0.0.0.0:5000 app:app
```

2. **Add Nginx Reverse Proxy**:

```nginx
server {
    listen 80;
    client_max_body_size 20M;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
```

3. **Add Authentication**: Implement API key authentication for security

4. **Add Rate Limiting**: Prevent abuse with request rate limiting

5. **Add Logging**: Configure proper logging for monitoring

## Error Handling

The API returns appropriate HTTP status codes:

- `200`: Success
- `400`: Bad request (invalid files, no images provided)
- `500`: Internal server error

Error responses include JSON with error descriptions:

```json
{
	"error": "No images provided"
}
```

## File Structure

```
.
├── app.py              # Main Flask application
├── requirements.txt    # Python dependencies
├── test_api.py        # API testing script
├── start_server.sh    # Server startup script
├── yolov8_model.pt    # YOLO model weights
├── uploads/           # Temporary upload directory
└── README.md          # This file
```

## Dependencies

- Flask: Web framework
- PyTorch: Deep learning framework
- Ultralytics: YOLO implementation
- Transformers: Hugging Face transformers for LLM
- OpenCV: Image processing
- FPDF2: PDF generation
- Matplotlib: Visualization for PDF reports

## Troubleshooting

### Common Issues

1. **Model not found**: Ensure `yolov8_model.pt` exists in the project directory
2. **CUDA out of memory**: Reduce batch size or use CPU-only mode
3. **Import errors**: Install all requirements with `pip install -r requirements.txt`
4. **Port already in use**: Change port in `app.py` or kill existing process

### Logs

Check console output for detailed error messages and model loading status.
