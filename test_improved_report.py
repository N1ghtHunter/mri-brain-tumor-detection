import json
import os
from datetime import datetime
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import glob

# Load the latest detection results
def load_latest_results():
    """Load the most recent tumor detection results.
    
    Returns:
        dict: The detection results data
    """
    result_files = glob.glob("detection_results_*.json")
    if not result_files:
        print("No detection result files found.")
        return None
    
    # Get the most recent file by modification time
    latest_file = max(result_files, key=os.path.getmtime)
    print(f"Loading latest detection results from: {latest_file}")
    
    try:
        with open(latest_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading detection results: {str(e)}")
        return None

# Medical knowledge module
def get_tumor_info_for_prompt(detected_types):
    """
    Get relevant medical information for the detected tumor types to include in prompts.
    """
    # Dictionary of common brain tumor types and their characteristics
    TUMOR_CHARACTERISTICS = {
        "glioma": {
            "description": "A type of tumor that starts in the glial cells of the brain",
            "types": ["Low-grade glioma", "High-grade glioma (Glioblastoma)"],
            "features": [
                "Often infiltrative (no clear borders)",
                "May show heterogeneous enhancement",
                "May have surrounding edema",
                "Can cross midline (butterfly glioma)"
            ],
            "implications": [
                "May cause seizures, headaches, or focal neurological deficits",
                "Treatment depends on grade, location, and genetic markers",
                "May require surgical resection, radiation, and/or chemotherapy"
            ]
        },
        "meningioma": {
            "description": "A tumor that forms on the membranes that cover the brain and spinal cord",
            "types": ["Benign meningioma", "Atypical meningioma", "Malignant meningioma"],
            "features": [
                "Usually well-defined with clear borders",
                "Often attached to dura mater",
                "Homogeneous enhancement",
                "May show a 'dural tail' sign"
            ],
            "implications": [
                "Often slow-growing and benign",
                "May cause symptoms from pressure on adjacent structures",
                "Surgical resection is often curative for accessible tumors"
            ]
        },
        "pituitary": {
            "description": "A tumor that develops in the pituitary gland at the base of the brain",
            "types": ["Microadenoma (<10mm)", "Macroadenoma (≥10mm)"],
            "features": [
                "Located in the sella turcica",
                "May extend to suprasellar region",
                "Can be functional (hormone-secreting) or non-functional",
                "Usually well-circumscribed"
            ],
            "implications": [
                "May cause hormonal imbalances",
                "Can lead to visual disturbances if pressing on optic chiasm",
                "Treatment options include medication, surgery, or radiation"
            ]
        },
        "no tumor": {
            "description": "No tumor detected in the brain MRI scan",
            "types": ["Normal brain scan"],
            "features": [
                "Normal brain anatomy",
                "No abnormal enhancements",
                "No mass effect",
                "No midline shift"
            ],
            "implications": [
                "Symptoms may be due to non-neoplastic causes",
                "Consider other neurological conditions",
                "Follow-up imaging may be warranted if symptoms persist"
            ]
        }
    }
    
    info = "REFERENCE INFORMATION FOR TUMOR TYPES:\n\n"
    
    for tumor_type in detected_types:
        tumor_type = tumor_type.lower()
        if tumor_type in TUMOR_CHARACTERISTICS:
            tumor_info = TUMOR_CHARACTERISTICS[tumor_type]
            info += f"{tumor_type.upper()}:\n"
            info += f"Description: {tumor_info['description']}\n"
            info += f"Common types: {', '.join(tumor_info['types'])}\n"
            info += "Typical features:\n"
            for feature in tumor_info['features']:
                info += f"- {feature}\n"
            info += "Clinical implications:\n"
            for implication in tumor_info['implications']:
                info += f"- {implication}\n"
            info += "\n"
    
    info += "Use this reference information to inform your medical report, but integrate it naturally."
    return info

def create_medical_prompt(results):
    """Create a more medically-focused prompt for the language model."""
    if not results:
        return "No detection results available."
    
    # Extract detected tumor types for medical knowledge
    detected_types = set()
    for image in results['images']:
        for detection in image['detections']:
            detected_types.add(detection['class'])
    
    # Get medical knowledge for the detected types
    medical_knowledge = get_tumor_info_for_prompt(list(detected_types))
    
    # Start with detailed instructions for the model
    prompt = """You are a radiologist specialized in brain MRI analysis. Write a detailed medical report in plain text based on the tumor detection results below. 
    
    IMPORTANT INSTRUCTIONS:
    1. DO NOT generate any code, Python syntax, or code blocks. No imports, function definitions, or variables.
    2. Write ONLY plain text in the style of a professional medical report.
    3. Format your response as a standard medical document.
    4. Never include triple backticks (```) or any programming syntax.
    
    The report should include:
    1. A formal header with 'RADIOLOGICAL REPORT' as the title
    2. A summary of findings
    3. Detailed description of each detected tumor (type, location, size)
    4. Potential clinical implications
    5. Recommendations for further tests or treatment
    6. Comparison with typical characteristics of each tumor type
    
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
    
    # Add medical knowledge if available
    if medical_knowledge:
        prompt += f"\n{medical_knowledge}\n"
    
    prompt += """\nWrite a comprehensive radiological report in formal medical language.

FORMAT REMINDER: Your output should look like this:

RADIOLOGICAL REPORT

Patient: [Anonymous]
Date of Examination: [Current Date]
Procedure: Brain MRI

FINDINGS:
[Your detailed findings here]

IMPRESSION:
[Your medical impression here]

RECOMMENDATIONS:
[Your recommendations here]

DO NOT include any code or programming syntax in your response.
Your task is ONLY to write a formal medical report based on the data provided."""
    return prompt

def load_small_llm(model_name="microsoft/phi-2", use_gpu=True):
    """Load a small, efficient language model for report generation."""
    print(f"Loading {model_name}...")
    
    # Set device
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
        device_map=device,
        trust_remote_code=True
    )
    
    return model, tokenizer

def generate_medical_report(model, tokenizer, prompt, max_length=1024, temperature=0.7):
    """Generate a medical report using the provided language model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate text
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=max_length,
            temperature=temperature,
            top_p=0.9,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id
        )
    
    # Decode the generated text
    report = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract only the newly generated content (after the prompt)
    if report.startswith(prompt):
        report = report[len(prompt):].strip()
    
    return report

def post_process_report(report):
    """Post-process the generated report to remove any code-like content."""
    # Remove code block markers
    report = report.replace('```python', '').replace('```', '')
    
    # Check for common code indicators
    code_indicators = ['import ', 'def ', 'class ', '# ', 'print(', '.py', 
                      'return ', 'if __name__', 'for ', 'while ', 'try:',
                      'with open', 'model.', 'tokenizer.']
    
    # If code indicators are found, apply more aggressive cleaning
    if any(indicator in report for indicator in code_indicators):
        print("Code detected in report. Applying cleanup...")
        
        # Split into lines and filter out code lines
        lines = report.split('\n')
        cleaned_lines = []
        
        for line in lines:
            # Skip lines that look like code
            if any(indicator in line for indicator in code_indicators):
                continue
            # Skip import statements and function definitions
            if line.strip().startswith(('import', 'from ', 'def ', 'class ')):
                continue
            # Keep lines that look like medical text
            if any(term in line.lower() for term in ['tumor', 'mri', 'brain', 'patient', 'finding', 'image', 'radiolog', 'recommend']):
                cleaned_lines.append(line)
            # Keep section headers
            elif line.strip().endswith(':') and len(line.strip()) < 50:
                cleaned_lines.append(line)
            # Keep short lines that don't look like code
            elif len(line.strip()) > 0 and len(line.strip()) < 100 and not any(c in line for c in '{}[]()=#'):
                cleaned_lines.append(line)
        
        # If we lost too much content, use a fallback approach
        if len(cleaned_lines) < 5:
            return create_fallback_report(report)
            
        report = '\n'.join(cleaned_lines)
    
    # Ensure the report starts with a proper header
    if not report.strip().startswith('RADIOLOGICAL REPORT'):
        report = 'RADIOLOGICAL REPORT\n\n' + report
    
    return report

def create_fallback_report(original_text):
    """Create a fallback report template when code generation is detected."""
    # Extract any useful medical terms from the original text
    medical_terms = []
    for term in ['tumor', 'glioma', 'meningioma', 'pituitary', 'lesion', 'mass', 'mm', 'size', 'location']:
        if term in original_text.lower():
            # Find sentences containing this term
            sentences = [s for s in original_text.split('.') if term in s.lower()]
            medical_terms.extend(sentences)
    
    # Create a basic report template
    report = """RADIOLOGICAL REPORT

Patient: [Anonymous]
Date of Examination: {}
Procedure: Brain MRI

FINDINGS:
Brain MRI demonstrates the presence of abnormal tissue consistent with tumor(s) as detected by the automated system.
""".format(datetime.now().strftime('%Y-%m-%d'))
    
    # Add any extracted medical terms
    if medical_terms:
        report += "\nNotes from analysis:\n"
        for term in medical_terms[:5]:  # Limit to 5 terms to avoid potential code snippets
            clean_term = term.strip().replace('\n', ' ')
            # Only add if it looks like a legitimate medical note (not code)
            if len(clean_term) > 10 and len(clean_term) < 200 and not any(c in clean_term for c in '{}[]()=#'):
                report += "- " + clean_term + ".\n"
    
    report += """
IMPRESSION:
The findings are suggestive of intracranial tumor(s). Clinical correlation is recommended.

RECOMMENDATIONS:
1. Clinical correlation with patient symptoms and history.
2. Consider follow-up imaging in 3-6 months.
3. Neurosurgical consultation may be warranted.
4. Consider additional advanced imaging techniques for further characterization."""
    
    return report

# Save report text to a file
def save_report_text(report_text, timestamp=None):
    """Save report text to a file."""
    if not timestamp:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    else:
        timestamp = timestamp.replace(" ", "_").replace(":", "")
        
    report_filename = f"medical_report_{timestamp}.txt"
    
    with open(report_filename, 'w') as f:
        f.write(f"MEDICAL REPORT - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(report_text)
    
    print(f"\nMedical report saved to {report_filename}")
    return report_filename

# Function to generate improved report
def generate_improved_report():
    """Generate medical report trying different models for better quality."""
    try:
        print("Generating improved medical report...")
        
        # Load detection data
        detection_data = load_latest_results()
        if not detection_data:
            return "No detection results available."
        
        # Create enhanced medical prompt
        medical_prompt = create_medical_prompt(detection_data)
        
        # Add extra instructions to prevent code generation
        medical_prompt = medical_prompt + "\n\nFINAL REMINDER: Your output MUST be a formal medical report in PLAIN TEXT ONLY. NEVER include any programming code, Python syntax, or code blocks. DO NOT try to process the data algorithmically. Simply write a radiologist's report based on the data provided.\n"
        
        # Try with default model - Phi-2
        print("Trying with microsoft/phi-2 model...")
        try:
            model, tokenizer = load_small_llm(model_name="microsoft/phi-2")
            
            # Generate with slightly higher temperature for more creativity
            raw_report = generate_medical_report(model, tokenizer, medical_prompt, temperature=0.75)
            
            # Post-process to remove any code-like content
            clean_report = post_process_report(raw_report)
            
            # Save the improved report
            timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_report_text(clean_report, timestamp_str)
            
            return clean_report
        except Exception as e:
            print(f"Error with model: {str(e)}")
            return f"Error generating report: {str(e)}"
    
    except Exception as e:
        print(f"Error generating improved report: {str(e)}")
        return f"Error: {str(e)}"

# Run the improved report generation
try:
    print("\nGenerating improved medical report with enhanced prompt...")
    improved_report = generate_improved_report()
    print("\nIMPROVED MEDICAL REPORT:\n")
    print(improved_report)
except Exception as e:
    print(f"Error: {str(e)}")
