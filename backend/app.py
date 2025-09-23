import os
from flask import Flask, request, jsonify
from search import get_image_features, find_similar_products, dummy_database

app = Flask(__name__)

# Create an 'uploads' directory to save the uploaded images
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return jsonify({"message": "Welcome to the Stuffcenter Backend API!"})

@app.route('/search', methods=['POST'])
def search():
    if 'image' not in request.files:
        return jsonify({"error": "No image file provided"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Get features from the uploaded image
        query_features = get_image_features(filepath)

        if query_features is None:
            return jsonify({"error": "Could not process image"}), 500

        # Find similar products (using our dummy database)
        search_results = find_similar_products(query_features, dummy_database)

        # Return the top 3 results
        top_results = search_results[:3]

        return jsonify({"results": [item['details'] for item in top_results]})

if __name__ == '__main__':
    app.run(debug=True)  