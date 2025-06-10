from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tempfile
import json
import glob
from datetime import datetime
import numpy as np
import cv2
import torch
from ultralytics import YOLO
from transformers import AutoModelForCausalLM, AutoTokenizer
from fpdf import FPDF
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Enable CORS for all routes
CORS(app)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH

# Create upload directory if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global variables to store loaded models (loaded once on startup)
yolo_model = None
llm_model = None
llm_tokenizer = None
classes = ["Glioma", "Meningioma", "No Tumor", "Pituitary"]

def load_latest_results():
    """Load the most recent detection results file."""
    json_files = glob.glob("detection_results_*.json")
    if not json_files:
        return None
    
    # Sort by modification time (most recent first)
    latest_file = max(json_files, key=os.path.getmtime)
    
    with open(latest_file, 'r') as f:
        results = json.load(f)
    
    return results

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load YOLO and LLM models on startup."""
    global yolo_model, llm_model, llm_tokenizer
    
    try:
        logger.info("Loading YOLO model...")
        yolo_model = YOLO('./yolov8_model.pt')
        logger.info("YOLO model loaded successfully")
        
        logger.info("Loading LLM model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        llm_tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2", trust_remote_code=True)
        llm_model = AutoModelForCausalLM.from_pretrained(
            "microsoft/phi-2",
            torch_dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map=device,
            trust_remote_code=True
        )
        logger.info("LLM model loaded successfully")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}")
        raise

def process_image(image_path):
    """Process a single image and return detection results."""
    try:
        # Load and process the image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Run YOLO inference
        results = yolo_model.predict(img_rgb, conf=0.25)[0]
        
        # Create image result dictionary
        image_result = {
            "image_path": image_path,
            "filename": os.path.basename(image_path),
            "detections": []
        }
        
        # Process detections
        if results.boxes is not None:
            for detection in results.boxes:
                x1, y1, x2, y2 = detection.xyxy[0].cpu().numpy()
                conf = detection.conf[0].cpu().numpy()
                cls = int(detection.cls[0].cpu().numpy())
                
                detection_info = {
                    "class": classes[cls],
                    "confidence": float(conf),
                    "bbox": {
                        "x1": float(x1),
                        "y1": float(y1),
                        "x2": float(x2),
                        "y2": float(y2),
                        "width": float(x2-x1),
                        "height": float(y2-y1)
                    }
                }
                image_result["detections"].append(detection_info)
        
        return image_result
    
    except Exception as e:
        logger.error(f"Error processing image {image_path}: {str(e)}")
        raise

def create_medical_prompt(results):
    """Create a medical prompt for the language model."""
    if not results or not results.get('images'):
        return "No detection results available."
    
    prompt = """IMPORTANT: DO NOT GENERATE CODE. WRITE A MEDICAL REPORT ONLY.

You are a radiologist specialized in brain MRI analysis. Write a detailed medical report in natural language based on the tumor detection results below. Do not include any code snippets or programming examples.
    
Your report should include:
1. A formal header with patient scan information
2. A summary of findings
3. Detailed description of each detected tumor (type, location, size)
4. Potential clinical implications
5. Recommendations for further tests or treatment
6. Comparison with typical characteristics of each tumor type

Structure the report as a formal medical document with appropriate sections and medical terminology.

Detection Results:\n"""
    
    prompt += f"Scan Date: {results['timestamp']}\n\n"
    
    for i, image in enumerate(results['images'], 1):
        prompt += f"Image {i}: {image['filename']}\n"
        
        if not image['detections']:
            prompt += "  No tumors detected.\n\n"
            continue
        
        for j, detection in enumerate(image['detections'], 1):
            prompt += f"  Tumor {j}:\n"
            prompt += f"    Type: {detection['class']}\n"
            prompt += f"    Confidence: {detection['confidence']:.2f}\n"
            
            # Calculate size in mm (assuming pixel-to-mm conversion factor)
            width_mm = detection['bbox']['width'] * 0.5  # Example conversion factor
            height_mm = detection['bbox']['height'] * 0.5  # Example conversion factor
            
            prompt += f"    Size: {width_mm:.1f}mm × {height_mm:.1f}mm\n"
            prompt += f"    Location: Region coordinates ({detection['bbox']['x1']:.1f}, {detection['bbox']['y1']:.1f})\n"
        
        prompt += "\n"
    
    prompt += "\nBased on these findings, write a comprehensive radiological report in the format of a standard medical document. DO NOT GENERATE CODE.\n"
    return prompt

def generate_medical_report(prompt, max_new_tokens=512, temperature=0.0):
    """Generate a medical report using the loaded LLM."""
    try:
        device = next(llm_model.parameters()).device
        inputs = llm_tokenizer(prompt, return_tensors="pt").to(device)
        
        do_sample = temperature > 0.0
        
        with torch.no_grad():
            output_ids = llm_model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=0.9 if do_sample else None,
                do_sample=do_sample,
                pad_token_id=llm_tokenizer.eos_token_id,
                eos_token_id=llm_tokenizer.eos_token_id,
                early_stopping=True
            )
        
        # Extract only the generated text
        gen_ids = output_ids[0, inputs["input_ids"].shape[-1]:]
        report_text = llm_tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        
        # Add timestamp header
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        formatted_report = f"MEDICAL REPORT - {timestamp}\n\n{report_text}"
        
        return formatted_report
    
    except Exception as e:
        logger.error(f"Error generating medical report: {str(e)}")
        raise

def create_pdf_report(detection_data, report_text):
    """Generate a PDF report with detection results and medical analysis."""
    try:
        # Create temporary PDF file
        pdf_fd, pdf_path = tempfile.mkstemp(suffix='.pdf')
        os.close(pdf_fd)
        
        pdf = FPDF()
        pdf.add_page()
        
        # Add title
        pdf.set_font("Arial", "B", 16)
        pdf.cell(0, 10, "Brain MRI Tumor Analysis Report", 0, 1, "C")
        pdf.set_font("Arial", "", 10)
        pdf.cell(0, 10, f"Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1, "C")
        pdf.ln(10)
        
        # Add detection summary
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Detection Summary", 0, 1, "L")
        pdf.set_font("Arial", "", 10)
        
        total_detections = sum(len(img['detections']) for img in detection_data['images'])
        pdf.cell(0, 10, f"Total images analyzed: {len(detection_data['images'])}", 0, 1, "L")
        pdf.cell(0, 10, f"Total detections: {total_detections}", 0, 1, "L")
        pdf.ln(10)
        
        # Add images with detection results
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Detection Images", 0, 1, "L")
        pdf.ln(5)
        
        temp_image_files = []
        
        try:
            # Process each image and add to PDF
            for i, image_data in enumerate(detection_data["images"]):
                # Load the original image
                img_path = image_data["image_path"]
                img = cv2.imread(img_path)
                
                if img is not None:
                    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    
                    # Create matplotlib figure
                    plt.figure(figsize=(10, 8))
                    plt.imshow(img_rgb)
                    
                    # Draw bounding boxes for detections
                    for detection in image_data["detections"]:
                        x1, y1 = detection["bbox"]["x1"], detection["bbox"]["y1"]
                        width, height = detection["bbox"]["width"], detection["bbox"]["height"]
                        cls = detection["class"]
                        conf = detection["confidence"]
                        
                        # Create rectangle for bounding box
                        rect = patches.Rectangle(
                            (x1, y1), width, height, 
                            linewidth=3, edgecolor='red', facecolor='none'
                        )
                        plt.gca().add_patch(rect)
                        
                        # Add text label with background
                        plt.text(
                            x1, y1-10, 
                            f"{cls} {conf:.2f}", 
                            color='white', fontsize=12, weight='bold',
                            bbox=dict(facecolor='red', alpha=0.8, pad=3)
                        )
                    
                    plt.title(f"Image {i+1}: {image_data['filename']}", fontsize=14, weight='bold')
                    plt.axis('off')
                    plt.tight_layout()
                    
                    # Save the figure to a temporary file
                    temp_img_fd, temp_img_path = tempfile.mkstemp(suffix='.png')
                    os.close(temp_img_fd)
                    plt.savefig(temp_img_path, dpi=150, bbox_inches='tight', facecolor='white')
                    plt.close()
                    
                    temp_image_files.append(temp_img_path)
                    
                    # Add image info to PDF
                    pdf.set_font("Arial", "B", 11)
                    pdf.cell(0, 10, f"Image {i+1}: {image_data['filename']}", 0, 1, "L")
                    pdf.set_font("Arial", "", 10)
                    
                    # Add detection details
                    if not image_data["detections"]:
                        pdf.cell(0, 10, "  No tumors detected", 0, 1, "L")
                    else:
                        for j, detection in enumerate(image_data["detections"]):
                            tumor_type = detection["class"]
                            confidence = detection["confidence"]
                            width_mm = detection['bbox']['width'] * 0.5
                            height_mm = detection['bbox']['height'] * 0.5
                            
                            pdf.cell(0, 10, f"  Detection {j+1}: {tumor_type} (Confidence: {confidence:.2f})", 0, 1, "L")
                            pdf.cell(0, 10, f"    Size: {width_mm:.1f}mm × {height_mm:.1f}mm", 0, 1, "L")
                    
                    pdf.ln(5)
                    
                    # Add the image to PDF (adjust size to fit page)
                    try:
                        # Calculate image dimensions to fit on page
                        page_width = pdf.w - 40  # Leave 20mm margin on each side
                        max_height = 120  # Maximum height in mm
                        
                        pdf.image(temp_img_path, x=20, y=pdf.get_y(), w=page_width, h=max_height)
                        pdf.ln(max_height + 10)  # Move past the image
                        
                        # Add page break if not the last image
                        if i < len(detection_data["images"]) - 1:
                            pdf.add_page()
                            
                    except Exception as img_error:
                        logger.warning(f"Could not add image to PDF: {str(img_error)}")
                        pdf.cell(0, 10, f"  (Image could not be displayed: {str(img_error)})", 0, 1, "L")
                        pdf.ln(5)
                
                else:
                    pdf.set_font("Arial", "", 10)
                    pdf.cell(0, 10, f"Image {i+1}: Could not load {image_data['filename']}", 0, 1, "L")
                    pdf.ln(5)
        
        except Exception as e:
            logger.error(f"Error processing images for PDF: {str(e)}")
            pdf.cell(0, 10, f"Error processing images: {str(e)}", 0, 1, "L")
        
        # Add medical report on new page
        pdf.add_page()
        pdf.set_font("Arial", "B", 12)
        pdf.cell(0, 10, "Medical Analysis Report", 0, 1, "L")
        pdf.ln(5)
        
        # Add report text
        pdf.set_font("Arial", "", 10)
        lines = report_text.split('\n')
        for line in lines:
            if line.strip().endswith(':') and len(line.strip()) < 50:
                pdf.set_font("Arial", "B", 11)
                pdf.cell(0, 10, line.strip(), 0, 1, "L")
                pdf.set_font("Arial", "", 10)
            else:
                # Handle long lines by wrapping them
                if line.strip():
                    pdf.multi_cell(0, 5, line.strip())
                    pdf.ln(2)
        
        # Save PDF
        pdf.output(pdf_path)
        
        # Clean up temporary image files
        for temp_file in temp_image_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
            except Exception as e:
                logger.warning(f"Could not remove temp image file {temp_file}: {str(e)}")
        
        return pdf_path
    
    except Exception as e:
        logger.error(f"Error creating PDF report: {str(e)}")
        raise

@app.route('/')
def index():
    """Serve the frontend HTML page."""
    return send_file('frontend.html')

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'models_loaded': {
            'yolo': yolo_model is not None,
            'llm': llm_model is not None and llm_tokenizer is not None
        },
        'timestamp': datetime.now().isoformat()
    })

@app.route('/analyze', methods=['POST'])
def analyze_images():
    """Main endpoint to analyze images and return PDF report."""
    try:
        # Check if files were uploaded
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        files = request.files.getlist('images')
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No images selected'}), 400
        
        # Process uploaded images
        detection_results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "images": []
        }
        
        temp_files = []
        
        try:
            for file in files:
                if file and allowed_file(file.filename):
                    # Save uploaded file temporarily
                    filename = secure_filename(file.filename)
                    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                           f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
                    file.save(temp_path)
                    temp_files.append(temp_path)
                    
                    # Process the image
                    image_result = process_image(temp_path)
                    detection_results["images"].append(image_result)
            
            if not detection_results["images"]:
                return jsonify({'error': 'No valid images to process'}), 400
            
            # Generate medical report
            logger.info("Generating medical report...")
            medical_prompt = create_medical_prompt(detection_results)
            report_text = generate_medical_report(medical_prompt)
            
            # Create PDF
            logger.info("Creating PDF report...")
            pdf_path = create_pdf_report(detection_results, report_text)
            
            # Return PDF file
            return send_file(
                pdf_path,
                as_attachment=True,
                download_name=f"brain_tumor_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                mimetype='application/pdf'
            )
        
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Could not remove temp file {temp_file}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in analyze_images: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

@app.route('/analyze-json', methods=['POST'])
def analyze_images_json():
    """Alternative endpoint that returns JSON results instead of PDF."""
    try:
        # Similar to analyze_images but returns JSON
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided'}), 400
        
        files = request.files.getlist('images')
        if not files or all(file.filename == '' for file in files):
            return jsonify({'error': 'No images selected'}), 400
        
        detection_results = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "images": []
        }
        
        temp_files = []
        
        try:
            for file in files:
                if file and allowed_file(file.filename):
                    filename = secure_filename(file.filename)
                    temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 
                                           f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{filename}")
                    file.save(temp_path)
                    temp_files.append(temp_path)
                    
                    image_result = process_image(temp_path)
                    detection_results["images"].append(image_result)
            
            if not detection_results["images"]:
                return jsonify({'error': 'No valid images to process'}), 400
            
            # Generate medical report
            medical_prompt = create_medical_prompt(detection_results)
            report_text = generate_medical_report(medical_prompt)
            
            return jsonify({
                'detection_results': detection_results,
                'medical_report': report_text,
                'summary': {
                    'total_images': len(detection_results['images']),
                    'total_detections': sum(len(img['detections']) for img in detection_results['images'])
                }
            })
        
        finally:
            # Clean up temporary files
            for temp_file in temp_files:
                try:
                    if os.path.exists(temp_file):
                        os.remove(temp_file)
                except Exception as e:
                    logger.warning(f"Could not remove temp file {temp_file}: {str(e)}")
    
    except Exception as e:
        logger.error(f"Error in analyze_images_json: {str(e)}")
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500

if __name__ == '__main__':
    # Load models on startup
    logger.info("Starting application and loading models...")
    load_models()
    logger.info("Models loaded successfully. Starting Flask server...")
    
    # Run the app
    app.run(debug=False, host='0.0.0.0', port=5000)
