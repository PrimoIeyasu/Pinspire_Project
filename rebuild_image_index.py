import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
from pathlib import Path

# Set the Django settings module environment variable
os.environ['DJANGO_SETTINGS_MODULE'] = 'pinterestClone.settings'

# Initialize Django
import django
django.setup()

from pins.models import Pin

BASE_DIR = Path(__file__).resolve().parent
media_url = os.path.join(BASE_DIR, 'media')

# Verify specific image files
specific_files = [
    'cat.jpg',
    'pexels-amar-preciado-10820109.jpg',
    'img.jpeg',
    'PedroPenduko1.webp',
    'PedroPenduko1_jmXuJx3.webp',
    'cat_kFqo4NU.jpg'
]

print("Verifying specific files in the media directory:")
for file in specific_files:
    image_path = os.path.join(media_url, file)
    try:
        with open(image_path, 'rb') as f:
            img = Image.open(f)
            img.verify()
            print(f"Successfully opened the image: {image_path}")
    except FileNotFoundError:
        print(f"File not found: {image_path}")
    except Exception as e:
        print(f"Error: {e}")

# Load the model
model = SentenceTransformer('clip-ViT-B-32')

# Create the underlying FAISS index
index = faiss.IndexFlatL2(512)  # Adjust dimension if needed

# Create the ID map to enable adding with IDs
image_index = faiss.IndexIDMap(index)

# Fetch all pins and their images
pins = Pin.objects.all()

# Print raw pin image values for debugging
print("Raw Pin Image Values:")
for pin in pins:
    print(f"Pin ID: {pin.id}, Raw Image Value: {pin.image}")

# Construct image paths correctly
image_paths = [os.path.abspath(os.path.join(media_url, str(pin.image))) for pin in pins]
pin_ids = [pin.id for pin in pins]

# Debugging: Print image paths to verify
print("Image Paths:")
for pin, path in zip(pins, image_paths):
    print(f"Pin ID: {pin.id}, Image Path: {path}")

# Encode the images
image_embeddings = []
for path in image_paths:
    try:
        # Test if the file can be opened
        with open(path, 'rb') as f:
            img = Image.open(f)
            img.verify()  # Verify that it is, in fact, an image

        img = Image.open(path)
        embedding = model.encode([img])[0]
        image_embeddings.append(embedding)
        print(f"Successfully opened and encoded: {path}")
    except (FileNotFoundError, IOError) as e:
        print(f"Error opening {path}: {e}")
        continue

# Debugging: Verify the embeddings and their dimensions
image_embeddings = np.array(image_embeddings)
print(f"Total images: {len(image_paths)}")
print(f"Total embeddings: {image_embeddings.shape}")
if len(image_embeddings) > 0:
    print(f"First embedding: {image_embeddings[0]}")

# Add embeddings to the index
if len(image_embeddings) > 0:
    image_index.add_with_ids(image_embeddings, np.array(pin_ids))

# Debugging: Verify the index
print(f"Index size: {image_index.ntotal}")

# Save the index to file
faiss.write_index(image_index, os.path.join(media_url, 'vector.index'))

print("Image index rebuilt and saved.")
