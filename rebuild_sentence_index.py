import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path

# Set the Django settings module environment variable
os.environ['DJANGO_SETTINGS_MODULE'] = 'pinterestClone.settings'

# Initialize Django
import django
django.setup()

from pins.models import Pin

BASE_DIR = Path(__file__).resolve().parent.parent
media_url = os.path.join(BASE_DIR, 'media')

# Load the model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Create the underlying FAISS index
index = faiss.IndexFlatL2(384)  # Adjust dimension if needed

# Create the ID map to enable adding with IDs
sentence_index = faiss.IndexIDMap(index)

# Fetch all pins and their names
pins = Pin.objects.all()
sentences = [pin.name for pin in pins]
pin_ids = [pin.id for pin in pins]

# Encode the sentences
sentence_embeddings = model.encode(sentences)

# Debugging: Verify the embeddings and their dimensions
print(f"Total sentences: {len(sentences)}")
print(f"Total embeddings: {sentence_embeddings.shape}")
print(f"First embedding: {sentence_embeddings[0]}")

# Add embeddings to the index
sentence_index.add_with_ids(np.array(sentence_embeddings), np.array(pin_ids))

# Debugging: Verify the index
print(f"Index size: {sentence_index.ntotal}")

# Save the index to file
faiss.write_index(sentence_index, os.path.join(media_url, 'sentence.index'))

print("Sentence index rebuilt and saved.")
