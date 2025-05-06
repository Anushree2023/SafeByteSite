from flask import Flask, render_template, request, redirect, url_for
import os
from werkzeug.utils import secure_filename
from model import ocr_to_text, clean_ingredients, load_model, analyze_ingredients_llm, format_results

app = Flask(__name__)

# Configuring upload folder and allowed extensions
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Function to check allowed file extensions
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

# Load the model once when the app starts
load_model()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Step 1: Perform OCR on the uploaded image
        text = ocr_to_text(filepath)
        
        # Step 2: Clean the ingredients from OCR text
        ingredients = clean_ingredients(text)
        
        # Step 3: Analyze ingredients using LLM model
        analysis_data = analyze_ingredients_llm(ingredients)
        
        # Step 4: Format and render the result
        result = format_results(analysis_data)
        
        return render_template('result.html', result=result, image_url=filepath)

    return redirect(url_for('home'))

if __name__ == '__main__':
    app.run(debug=True)
