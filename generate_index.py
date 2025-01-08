import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from PIL import Image
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
media_url = os.path.join(BASE_DIR, 'media')

# Ensure the 'media' directory exists
if not os.path.exists(media_url):
    os.makedirs(media_url)

# Load your images and create feature vectors
image_paths = ["path/to/your/image1.jpg", "path/to/your/image2.jpg", "path/to/your/image3.jpg"]  # Update with your image paths
model = SentenceTransformer('clip-ViT-B-32')

# Placeholder for embeddings
embeddings = []

for image_path in image_paths:
    img = Image.open(image_path)
    img_embedding = model.encode([img])
    embeddings.append(img_embedding)

# Convert to numpy array
embeddings = np.vstack(embeddings)

# Create and populate the FAISS index
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save the index to a file
index_file_path = os.path.join(media_url, 'vector.index')
faiss.write_index(index, index_file_path)

print(f"FAISS index created and saved to {index_file_path}")
