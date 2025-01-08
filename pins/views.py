import os
import re
import numpy as np
from django.http import JsonResponse
from rest_framework.views import APIView
from rest_framework.response import Response
from pins.models import Pin
from pins.serializers import pin_serializer
from sentence_transformers import SentenceTransformer
from PIL import Image
import faiss
from pathlib import Path
import random

BASE_DIR = Path(__file__).resolve().parent.parent

class Pins(APIView):
    def get(self, request):
        allPinsCount = Pin.objects.all().count()

        if allPinsCount <= 20:
            allPins = Pin.objects.all()
        else:
            num = random.randint(0, allPinsCount - 20)
            allPins = Pin.objects.all()[num:num + 20]

        all_responses_received = []

        for pin in allPins:
            image = str(pin.image)
            all_responses_received.append({
                'image': image, "slug": pin.slug,
                "name": pin.name, "id": pin.id
            })

        return JsonResponse(all_responses_received, safe=False)

    def post(self, request, *args, **kwargs):
        serializer = pin_serializer(data=request.data)
        if serializer.is_valid():
            name = serializer.validated_data.get('name').strip().lower()
            temporalSlug = name.strip().lower()
            temporal_name = re.sub("[$&+,;:=?@#|'<>.^*()%!\s+\"`]", "-", f'{temporalSlug}')
            new_product_slug = temporal_name + f'-{Pin.objects.all().count()}'
            pin = serializer.save(slug=new_product_slug)

            media_url = os.path.join(BASE_DIR, 'media')

            # Handle image vector index
            vector_index_file = f"{media_url}/vector.index"
            sentence_index_file = f"{media_url}/sentence.index"

            if os.path.exists(vector_index_file):
                index = faiss.read_index(vector_index_file)
            else:
                index = faiss.IndexFlatL2(512)  # Adjust dimension if needed

            model = SentenceTransformer('clip-ViT-B-32')

            image_path = f"{media_url}/{request.FILES['image'].name}"
            with open(image_path, 'wb+') as destination:
                for chunk in request.FILES['image'].chunks():
                    destination.write(chunk)

            img = Image.open(image_path)
            img_embedding = model.encode([img])
            
            # Use the pin ID for the embedding ID
            index.add_with_ids(np.array(img_embedding), np.array([pin.id]))

            faiss.write_index(index, vector_index_file)

            # Handle sentence index
            sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
            sentence_embedding = sentence_model.encode([name])

            if os.path.exists(sentence_index_file):
                sentence_index = faiss.read_index(sentence_index_file)
                if sentence_index.d != sentence_embedding.shape[1]:
                    raise ValueError(f"Dimension mismatch: sentence index has dimension {sentence_index.d}, but embedding has dimension {sentence_embedding.shape[1]}")
            else:
                sentence_index = faiss.IndexFlatL2(sentence_embedding.shape[1])

            sentence_index.add_with_ids(np.array(sentence_embedding), np.array([pin.id]))

            faiss.write_index(sentence_index, sentence_index_file)

            return Response({'status': 'saved'})
        else:
            return Response(serializer.errors)
