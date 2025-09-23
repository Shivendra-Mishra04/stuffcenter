import cv2
import numpy as np
from search import get_image_features, find_similar_products, dummy_database

# Create a dummy image file for testing
# This will create a temporary blue image.
dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
dummy_image[:, :, 0] = 255 # Set blue channel to max
cv2.imwrite('dummy_image.png', dummy_image)

# We will now create some dummy product images to add features to our database
blue_shirt_img = np.zeros((100, 100, 3), dtype=np.uint8)
blue_shirt_img[:, :, 0] = 255
cv2.imwrite('blue_shirt.png', blue_shirt_img)

red_shirt_img = np.zeros((100, 100, 3), dtype=np.uint8)
red_shirt_img[:, :, 2] = 255
cv2.imwrite('red_shirt.png', red_shirt_img)

yellow_shirt_img = np.zeros((100, 100, 3), dtype=np.uint8)
yellow_shirt_img[:, :, 2] = 255
yellow_shirt_img[:, :, 1] = 255
cv2.imwrite('yellow_shirt.png', yellow_shirt_img)

# Extract features and update the dummy database
dummy_database[0]['features'] = get_image_features('blue_shirt.png')
dummy_database[1]['features'] = get_image_features('red_shirt.png')
dummy_database[2]['features'] = get_image_features('yellow_shirt.png')

# Now, test the functions
print("Testing the get_image_features function...")
features = get_image_features('dummy_image.png')
if features is not None:
    print("Features extracted successfully.")
else:
    print("Error: Could not extract features.")

print("\nTesting the find_similar_products function...")
if features is not None:
    similar_products = find_similar_products(features, dummy_database)
    print("Search results (ordered from most to least similar):")
    for product in similar_products:
        print(f"Product: {product['details']['name']} | Similarity Score: {product['similarity']:.2f}")

print("\nTest complete.")
