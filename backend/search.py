import cv2
import numpy as np

def get_image_features(image_path):
    """
    Extracts features from an image using a simple color histogram.
    In a real-world app, you would use a more advanced CNN model here.
    """
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Convert image to a different color space (e.g., HSV) for better results
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Calculate a color histogram as a simple feature vector
    hist = cv2.calcHist([hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
    cv2.normalize(hist, hist)
    return hist.flatten()

def find_similar_products(query_features, product_database):
    """
    Finds the most similar products in a database based on their features.
    """
    # In a real-world app, this would be a more complex similarity search (e.g., using FAISS)
    # For this example, we'll just do a simple comparison.
    results = []
    for product in product_database:
        similarity = cv2.compareHist(
            np.float32(query_features),
            np.float32(product['features']),
            cv2.HISTCMP_CORREL
        )
        results.append({
            'id': product['id'],
            'similarity': similarity,
            'details': product['details']
        })

    # Sort results from most similar to least similar
    results.sort(key=lambda x: x['similarity'], reverse=True)
    return results

# This is a dummy database to test our functions.
# We will replace this with real data later.
dummy_database = [
    {'id': 1, 'details': {'name': 'Blue Shirt', 'price': 25.00}, 'features': None},
    {'id': 2, 'details': {'name': 'Red Shirt', 'price': 30.00}, 'features': None},
    {'id': 3, 'details': {'name': 'Yellow T-Shirt', 'price': 22.50}, 'features': None}
]