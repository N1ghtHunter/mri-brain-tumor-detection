"""
This module contains common definitions for tumor types that can be detected in brain MRI scans.
The information can be used to enhance language model prompts for medical report generation.
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
    }
}

# Function to get relevant medical information based on detected tumor types
def get_tumor_info_for_prompt(detected_types):
    """
    Get relevant medical information for the detected tumor types to include in prompts.
    
    Args:
        detected_types: List of detected tumor types
        
    Returns:
        str: Formatted string with relevant medical information
    """
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
