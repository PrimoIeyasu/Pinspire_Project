import os
import django
from pathlib import Path

# Set the environment variable for Django settings module
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'pinterestClone.settings')

# Initialize Django
django.setup()

from pins.models import Pin
from sentence_transformers import SentenceTransformer
from PIL import Image
import faiss
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
media_url = os.path.join(BASE_DIR, 'media')

# Create or read existing Faiss index
vector_index_file = f"{media_url}/vector.index"
sentence_index_file = f"{media_url}/sentence.index"

if os.path.exists(vector_index_file):
    vector_index = faiss.read_index(vector_index_file)
else:
    vector_index = faiss.IndexFlatL2(512)  # Adjust dimension if needed

if os.path.exists(sentence_index_file):
    sentence_index = faiss.read_index(sentence_index_file)
else:
    sentence_index = faiss.IndexFlatL2(384)

image_model = SentenceTransformer('clip-ViT-B-32')
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# Re-index all pins
for pin in Pin.objects.all():
    # Index image embeddings
    image_path = os.path.join(media_url, str(pin.image))
    if os.path.exists(image_path):
        img = Image.open(image_path)
        img_embedding = image_model.encode([img])
        vector_index.add_with_ids(np.array(img_embedding), np.array([pin.id]))
        print(f"Added image embedding for Pin ID: {pin.id}")
    
    # Index sentence embeddings
    sentence_embedding = sentence_model.encode([pin.name])
    sentence_index.add_with_ids(np.array(sentence_embedding), np.array([pin.id]))
    print(f"Added sentence embedding for Pin ID: {pin.id}")

# Save the indices
faiss.write_index(vector_index, vector_index_file)
print(f"Saved vector index to {vector_index_file}")
faiss.write_index(sentence_index, sentence_index_file)
print(f"Saved sentence index to {sentence_index_file}")

print("Re-indexing complete!")
