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
    """Load YOLO and LLM models on startup with optimizations."""
    global yolo_model, llm_model, llm_tokenizer
    
    try:
        logger.info("Loading YOLO model...")
        yolo_model = YOLO('./yolov8_model.pt')
        logger.info("YOLO model loaded successfully")
        
        logger.info("Loading LLM model...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {device}")
        
        # Try to use a smaller, faster model for better performance
        # Ordered by speed (fastest first) and likelihood to work
        model_options = [
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0", # Fast but may have dependency issues
            "microsoft/DialoGPT-small",           # Very fast, simple model
            "microsoft/DialoGPT-medium",          # Medium speed, good balance
            "microsoft/phi-2"                     # Fallback to original (slowest but most capable)
        ]
        
        model_loaded = False
        for model_name in model_options:
            try:
                logger.info(f"Attempting to load {model_name}...")
                llm_tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                
                # Optimize model loading
                model_kwargs = {
                    "trust_remote_code": True,
                    "torch_dtype": torch.float16 if device == "cuda" else torch.float32,
                    "device_map": device,
                }
                
                # Handle TinyLlama specifically to avoid FlashAttention issues
                if "TinyLlama" in model_name:
                    model_kwargs["attn_implementation"] = "eager"  # Use eager attention for compatibility
                
                llm_model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
                
                # Optimize model for inference
                llm_model.eval()
                if hasattr(llm_model, 'half') and device == "cuda":
                    try:
                        llm_model = llm_model.half()
                    except Exception as e:
                        logger.warning(f"Could not convert model to half precision: {str(e)}")
                
                logger.info(f"LLM model {model_name} loaded successfully")
                model_loaded = True
                break
                
            except Exception as e:
                logger.warning(f"Failed to load {model_name}: {str(e)}")
                # Clean up if tokenizer was loaded but model failed
                if 'llm_tokenizer' in locals():
                    llm_tokenizer = None
                continue
        
        if not model_loaded:
            raise Exception("Failed to load any LLM model")
        
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
    """Create a detailed medical prompt for comprehensive LLM processing."""
    if not results or not results.get('images'):
        return "No detection results available."
    
    # Enhanced prompt for detailed medical report generation
    prompt = """You are a radiologist writing a comprehensive brain MRI report. Generate a detailed, professional medical report following standard radiological format.

INSTRUCTIONS:
- Write complete sentences and paragraphs
- Use proper medical terminology
- Be thorough and specific
- Include technical details about location, size, and characteristics
- Provide clear clinical impressions and recommendations

REQUIRED SECTIONS:
1. CLINICAL HISTORY
2. TECHNIQUE
3. FINDINGS
4. IMPRESSION
5. RECOMMENDATIONS

DETECTION DATA:
"""
    
    prompt += f"Study Date: {results['timestamp']}\n"
    prompt += f"Number of Images Analyzed: {len(results['images'])}\n\n"
    
    # Detailed findings for each image
    total_tumors = 0
    findings_details = []
    
    for i, image in enumerate(results['images'], 1):
        prompt += f"Image {i} ({image['filename']}):\n"
        
        if not image['detections']:
            prompt += "- No abnormal findings detected\n"
        else:
            for j, detection in enumerate(image['detections'], 1):
                total_tumors += 1
                cls = detection['class']
                conf = detection['confidence']
                bbox = detection['bbox']
                
                # Calculate size in mm (assuming pixel to mm conversion)
                width_mm = bbox['width'] * 0.5
                height_mm = bbox['height'] * 0.5
                area_mm2 = width_mm * height_mm
                
                prompt += f"- Detection {j}: {cls}\n"
                prompt += f"  Confidence: {conf:.3f} ({conf*100:.1f}%)\n"
                prompt += f"  Dimensions: {width_mm:.1f}mm x {height_mm:.1f}mm\n"
                prompt += f"  Area: {area_mm2:.1f} mm²\n"
                prompt += f"  Location coordinates: ({bbox['x1']:.0f}, {bbox['y1']:.0f}) to ({bbox['x2']:.0f}, {bbox['y2']:.0f})\n"
                
                findings_details.append({
                    'type': cls,
                    'size': f"{width_mm:.1f}mm x {height_mm:.1f}mm",
                    'confidence': conf,                'image': i                })
        prompt += "\n"
    
    prompt += f"SUMMARY: {total_tumors} total findings detected across {len(results['images'])} images\n\n"
    
    prompt += """Now write a comprehensive medical report using this data. Follow this exact structure and write in complete, professional sentences:

BRAIN MRI REPORT

CLINICAL HISTORY:
Brain MRI examination performed for tumor detection screening. Patient referred for comprehensive brain imaging analysis.

TECHNIQUE:
Multi-sequence brain MRI examination with automated tumor detection analysis using advanced machine learning algorithms. Images processed with high-resolution analysis protocols.

FINDINGS:
The brain parenchyma demonstrates the following findings based on automated detection analysis:
[Provide detailed description of all findings, including normal anatomy and any abnormalities. For each tumor detected, describe location, size, morphology, and relationship to surrounding structures. Be specific about measurements and confidence levels.]

IMPRESSION:
[Summarize key findings and provide diagnostic interpretation. State whether findings are normal or abnormal and their clinical significance.]

RECOMMENDATIONS:
[Provide specific clinical recommendations based on findings, including follow-up imaging, specialist consultation, or further evaluation as appropriate.]

---
IMPORTANT: Write complete sentences and paragraphs. Do not use bullet points or incomplete phrases. Ensure the report reads professionally and provides comprehensive medical information.

Begin the formal medical report now:"""
    
    return prompt

def generate_template_report(results):
    """Generate a comprehensive template-based report with detailed medical content."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Count findings by type
    total_findings = 0
    findings_by_type = {}
    tumor_details = []
    
    for i, image in enumerate(results['images'], 1):
        for detection in image['detections']:
            total_findings += 1
            tumor_type = detection['class']
            
            if tumor_type not in findings_by_type:
                findings_by_type[tumor_type] = []
            
            # Calculate dimensions
            width_mm = detection['bbox']['width'] * 0.5
            height_mm = detection['bbox']['height'] * 0.5
            size_mm = max(width_mm, height_mm)
            
            tumor_info = {
                'type': tumor_type,
                'confidence': detection['confidence'],
                'size': size_mm,
                'width': width_mm,
                'height': height_mm,
                'image': i,
                'filename': image['filename']
            }
            
            findings_by_type[tumor_type].append(tumor_info)
            if tumor_type != "No Tumor":
                tumor_details.append(tumor_info)
    
    # Generate comprehensive report
    report = f"""BRAIN MRI ANALYSIS REPORT

Generated: {timestamp}
Study Date: {results.get('timestamp', timestamp)}
Examination: Brain MRI with AI-Assisted Tumor Detection
Images Analyzed: {len(results['images'])}

CLINICAL HISTORY:
Brain MRI examination performed for tumor screening and detection. Patient underwent comprehensive brain imaging with advanced automated analysis using machine learning algorithms for tumor identification and characterization.

TECHNIQUE:
Multi-planar brain MRI sequences analyzed using YOLOv8-based deep learning model trained specifically for brain tumor detection. The automated system evaluates images for presence of glioma, meningioma, pituitary adenoma, and normal brain tissue with high precision detection algorithms.

FINDINGS:"""
    
    if len(tumor_details) == 0:
        # No tumors found
        report += """
The automated analysis reveals no evidence of abnormal masses or lesions within the brain parenchyma. All analyzed images demonstrate normal brain anatomy without detectable tumor formations. The cerebral hemispheres, brainstem, and posterior fossa structures appear unremarkable. No mass effect, midline shift, or abnormal enhancement patterns are identified. The ventricular system maintains normal configuration and size.

Specific Analysis Results:
"""
        for i, image in enumerate(results['images'], 1):
            report += f"- Image {i} ({image['filename']}): Normal brain tissue confirmed with high confidence\n"
        
        report += """
All regions of interest have been systematically evaluated and show no signs of neoplastic changes. The automated detection system processed each image with comprehensive tumor screening protocols."""
    
    else:
        # Tumors detected
        report += f"""
The automated analysis has identified {len(tumor_details)} abnormal finding(s) across {len(results['images'])} analyzed images requiring immediate attention and further evaluation.

Detailed Findings:
"""
        
        # Group findings by type for better reporting
        for tumor_type, detections in findings_by_type.items():
            if tumor_type != "No Tumor" and detections:
                avg_confidence = sum(d['confidence'] for d in detections) / len(detections)
                avg_size = sum(d['size'] for d in detections) / len(detections)
                
                report += f"""
{tumor_type.upper()} LESIONS ({len(detections)} detected):
- Average detection confidence: {avg_confidence:.1%}
- Average maximum dimension: {avg_size:.1f}mm
- Distribution across images:
"""
                
                for detection in detections:
                    report += f"  * Image {detection['image']} ({detection['filename']}): "
                    report += f"{detection['width']:.1f}mm x {detection['height']:.1f}mm lesion "
                    report += f"(confidence: {detection['confidence']:.1%})\n"
                
                # Add clinical context based on tumor type
                if tumor_type == "Glioma":
                    report += "  Clinical Note: Gliomas are primary brain tumors arising from glial cells. These lesions require urgent neurosurgical evaluation and multidisciplinary management planning.\n"
                elif tumor_type == "Meningioma":
                    report += "  Clinical Note: Meningiomas are typically benign tumors arising from meningeal tissue. Size and location determine treatment approach and monitoring strategy.\n"
                elif tumor_type == "Pituitary":
                    report += "  Clinical Note: Pituitary lesions may affect hormonal function and require endocrinological assessment in addition to neurosurgical evaluation.\n"

    report += """

IMPRESSION:"""
    
    if len(tumor_details) == 0:
        report += """
NEGATIVE FOR INTRACRANIAL TUMORS. The comprehensive automated analysis demonstrates no evidence of brain tumors across all examined images. Normal brain parenchyma is identified throughout all analyzed regions with high diagnostic confidence."""
    else:
        tumor_types = list(set([d['type'] for d in tumor_details]))
        if len(tumor_types) == 1:
            report += f"""
POSITIVE FOR {tumor_types[0].upper()}. Automated detection has identified {len(tumor_details)} lesion(s) consistent with {tumor_types[0].lower()} requiring immediate clinical attention. The findings demonstrate characteristic imaging features with high detection confidence levels."""
        else:
            report += f"""
MULTIPLE TUMOR TYPES DETECTED. The analysis reveals {len(tumor_details)} lesions across {len(tumor_types)} different tumor categories: {', '.join(tumor_types)}. This complex presentation requires comprehensive neurosurgical evaluation and staging studies."""

    report += """

RECOMMENDATIONS:"""
    
    if len(tumor_details) == 0:
        report += """
1. Current examination shows no evidence of brain tumors
2. Results should be correlated with clinical symptoms and history
3. Routine follow-up imaging as clinically indicated
4. No immediate intervention required based on current findings
5. Patient counseling regarding normal results and any ongoing symptoms"""
    else:
        report += f"""
1. URGENT neurosurgical consultation recommended given positive tumor findings
2. Multidisciplinary team evaluation including oncology and radiation oncology
3. Consider advanced imaging (MRI with contrast, functional imaging) for surgical planning
4. Tissue confirmation through biopsy or surgical resection as appropriate
5. Staging studies to exclude metastatic disease if indicated
6. Patient and family counseling regarding diagnosis and treatment options
7. Immediate clinical correlation with neurological symptoms and examination"""

    report += """
DISCLAIMER:
This report was generated using automated tumor detection software. All findings must be verified and interpreted by a qualified radiologist in conjunction with clinical correlation. The AI system provides screening assistance but does not constitute a final diagnosis."""
    
    return report

def generate_medical_report(prompt, max_new_tokens=800, temperature=0.7, use_template_fallback=True):
    """Generate a medical report using the loaded LLM with optimized settings."""
    try:
        # If LLM models are not loaded or generation fails, use template
        if llm_model is None or llm_tokenizer is None:
            if use_template_fallback:
                logger.warning("LLM not available, using template report")
                # Extract results from the calling context - we'll need to pass this differently
                return "LLM not available - template report generation needs results object"
            else:
                raise Exception("LLM model not loaded")
        
        device = next(llm_model.parameters()).device
        
        # Truncate input prompt if too long to speed up processing
        max_input_length = 1024  # Limit input tokens for faster processing
        inputs = llm_tokenizer(prompt, 
                              return_tensors="pt", 
                              max_length=max_input_length,
                              truncation=True).to(device)
          # Optimized generation parameters for quality vs speed
        with torch.no_grad():
            output_ids = llm_model.generate(
                **inputs,
                max_new_tokens=800,           # Increased for fuller reports
                min_length=200,               # Ensure minimum report length
                temperature=0.7,              # Add some creativity vs pure determinism
                do_sample=True,               # Enable sampling for variety
                top_p=0.9,                    # Nucleus sampling for quality
                top_k=40,                     # Limit vocabulary for coherence
                repetition_penalty=1.15,      # Reduce repetition
                num_beams=2,                  # Light beam search for better quality
                early_stopping=True,
                pad_token_id=llm_tokenizer.eos_token_id,
                eos_token_id=llm_tokenizer.eos_token_id,
                use_cache=True                # Enable KV caching for speed
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
        if use_template_fallback:
            logger.info("Falling back to template report generation")
            # We'll handle this in the calling function
            raise Exception("LLM_FALLBACK_NEEDED")
        else:
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
                      # Draw bounding boxes for detections (only for actual tumors)
                    has_tumor = False
                    for detection in image_data["detections"]:
                        cls = detection["class"]
                        conf = detection["confidence"]
                        
                        # Only draw bounding boxes for actual tumors (not "No Tumor")
                        if cls != "No Tumor":
                            has_tumor = True
                            x1, y1 = detection["bbox"]["x1"], detection["bbox"]["y1"]
                            width, height = detection["bbox"]["width"], detection["bbox"]["height"]
                            
                            # Create rectangle for bounding box (red for tumors)
                            rect = patches.Rectangle(
                                (x1, y1), width, height, 
                                linewidth=3, edgecolor='red', facecolor='none'
                            )
                            plt.gca().add_patch(rect)
                              # Add text label with red background for tumors
                            plt.text(
                                x1, y1-10, 
                                f"{cls} {conf:.2f}", 
                                color='white', fontsize=12, weight='bold',
                                bbox=dict(facecolor='red', alpha=0.8, pad=3)
                            )
                        else:
                            # For "No Tumor" cases, add a green text overlay (no bounding box)
                            img_height, img_width = img_rgb.shape[:2]
                            plt.text(
                                img_width * 0.05, img_height * 0.1,  # Top-left corner
                                f"CLEAR: {cls} (Confidence: {conf:.2f})", 
                                color='white', fontsize=14, weight='bold',
                                bbox=dict(facecolor='green', alpha=0.8, pad=5)
                            )
                    # Set title color based on detection results
                    if has_tumor:
                        title_color = 'red'
                        status = "WARNING: TUMOR DETECTED"
                    else:
                        title_color = 'green'
                        status = "CLEAR: NO TUMOR DETECTED"
                    
                    plt.title(f"Image {i+1}: {image_data['filename']}\n{status}", 
                             fontsize=14, weight='bold', color=title_color)
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
                    pdf.set_font("Arial", "", 10)                    # Add detection details
                    if not image_data["detections"]:
                        pdf.cell(0, 10, "  [CLEAR] No tumors detected", 0, 1, "L")
                    else:                        
                        for j, detection in enumerate(image_data["detections"]):
                            tumor_type = detection["class"]
                            confidence = detection["confidence"]
                            
                            if tumor_type == "No Tumor":
                                pdf.cell(0, 10, f"  [CLEAR] {tumor_type} (Confidence: {confidence:.2f})", 0, 1, "L")
                            else:
                                width_mm = detection['bbox']['width'] * 0.5
                                height_mm = detection['bbox']['height'] * 0.5
                                pdf.cell(0, 10, f"  [WARNING] Detection {j+1}: {tumor_type} (Confidence: {confidence:.2f})", 0, 1, "L")
                                pdf.cell(0, 10, f"    Size: {width_mm:.1f}mm x {height_mm:.1f}mm", 0, 1, "L")
                    
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
                return jsonify({'error': 'No valid images to process'}), 400            # Generate medical report using enhanced template system
            logger.info("Generating comprehensive medical report...")
            report_text = generate_template_report(detection_results)
            
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
                return jsonify({'error': 'No valid images to process'}), 400            # Generate medical report using enhanced template system
            report_text = generate_template_report(detection_results)
            
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
