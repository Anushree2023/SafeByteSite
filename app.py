import os
from flask import Flask, request, render_template
from PIL import Image
import pytesseract

app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Fake Gemini classification logic
def classify_ingredients(text):
    ingredients = [i.strip().lower() for i in text.split(',')]
    results = []
    for ing in ingredients:
        if "sugar" in ing or "oil" in ing:
            results.append((ing, "Avoid"))
        elif "salt" in ing:
            results.append((ing, "Moderate"))
        elif ing:
            results.append((ing, "Safe"))
    return results

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/scan', methods=['POST'])
def scan():
    if 'image' not in request.files:
        return "No image uploaded.", 400

    image = request.files['image']
    path = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
    image.save(path)

    img = Image.open(path)
    extracted_text = pytesseract.image_to_string(img)

    # Classify the extracted text
    results = classify_ingredients(extracted_text)

    return render_template('result.html', results=results, image_path=path)

if __name__ == '__main__':
    app.run(debug=True)
