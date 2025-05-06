# model.py

import os
import cv2
import pytesseract
import symspellpy
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from symspellpy import SymSpell, Verbosity

# Initialize SymSpell for spell correction
sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
sym_spell.load_dictionary("frequency_dictionary_en_82_765.txt", term_index=0, count_index=1)

# Load pre-trained model for ingredient analysis
tokenizer = AutoTokenizer.from_pretrained("huggingface/mistral-7b-instruct-v0.3")
model = AutoModelForCausalLM.from_pretrained("huggingface/mistral-7b-instruct-v0.3")

# Function to perform OCR on the captured image and extract text
def ocr_to_text(image_path):
    """
    Perform Optical Character Recognition (OCR) to extract text from an image.
    
    Args:
    - image_path: Path to the image file to process.
    
    Returns:
    - extracted_text: Text extracted from the image.
    """
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Apply OCR to the image
    extracted_text = pytesseract.image_to_string(gray)
    return extracted_text

# Function to clean the extracted ingredients text
def clean_ingredients(extracted_text):
    """
    Clean the extracted ingredient text for further analysis.
    
    Args:
    - extracted_text: The raw text extracted from the image.
    
    Returns:
    - cleaned_text: Cleaned ingredient text.
    """
    # Example: Remove unwanted characters, make all lowercase
    cleaned_text = extracted_text.lower()
    cleaned_text = "".join(char if char.isalnum() or char.isspace() else " " for char in cleaned_text)
    return cleaned_text

# Function to load the model
def load_model():
    """
    Load the Hugging Face model for ingredient analysis.
    
    Returns:
    - model: Loaded Hugging Face model.
    - tokenizer: Corresponding tokenizer for the model.
    """
    # Load the pre-trained model
    model = AutoModelForCausalLM.from_pretrained("huggingface/mistral-7b-instruct-v0.3")
    tokenizer = AutoTokenizer.from_pretrained("huggingface/mistral-7b-instruct-v0.3")
    return model, tokenizer

# Function to analyze the ingredients using the model
def analyze_ingredients_llm(cleaned_text, model, tokenizer):
    """
    Analyze ingredients text using the Hugging Face language model.
    
    Args:
    - cleaned_text: The cleaned text to analyze.
    - model: The Hugging Face model used for analysis.
    - tokenizer: Tokenizer for the model.
    
    Returns:
    - analysis_result: The results of the analysis.
    """
    # Encode the text and pass it through the model
    inputs = tokenizer.encode(cleaned_text, return_tensors="pt")
    outputs = model.generate(inputs, max_length=200, num_return_sequences=1)
    decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return decoded_output

# Function to format the analysis results
def format_results(results):
    """
    Format the results into a structured format.
    
    Args:
    - results: The raw analysis results.
    
    Returns:
    - formatted_results: A structured format of the results (e.g., categories).
    """
    formatted_results = {
        'safe': [],
        'moderate': [],
        'avoid': []
    }

    # Assuming the results are a list of classifications or ingredient analysis
    for result in results:
        if "safe" in result.lower():
            formatted_results['safe'].append(result)
        elif "moderate" in result.lower():
            formatted_results['moderate'].append(result)
        elif "avoid" in result.lower():
            formatted_results['avoid'].append(result)
    
    return formatted_results
