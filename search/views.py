from django.http import JsonResponse, HttpResponse
import os
from rest_framework.parsers import JSONParser
from rest_framework.views import APIView
from rest_framework.response import Response
from pins.models import Pin
from sentence_transformers import SentenceTransformer
from PIL import Image
import faiss
from pathlib import Path
from asgiref.sync import async_to_sync, sync_to_async
from django.views.decorators.csrf import csrf_exempt
from django.core.exceptions import ObjectDoesNotExist
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent

class Search(APIView):
    def post(self, request, *args, **kwargs):
        if request.FILES.get("image") is not None:
            file = request.FILES['image']
            media_url = os.path.join(BASE_DIR, 'media')
            index = faiss.read_index(f"{media_url}/vector.index")
            model = SentenceTransformer('clip-ViT-B-32')
            img = Image.open(file)
            k = 8
            img_embedding = model.encode([img])
            
            # Create an array of unique IDs
            ids = np.arange(len(img_embedding))
            index.add_with_ids(np.array(img_embedding), ids)
            
            # Debug logging - print index data
            print(f"Index Data: {index}")

            D, I = index.search(img_embedding, k)
            all_responses_received = []
            for e in I[0]:
                if e != -1:
                    try:
                        pin = Pin.objects.get(id=e)
                        image = str(pin.image)
                        all_responses_received.append({
                            'image': image, 'name': pin.name,
                            'slug': pin.slug
                        })
                    except Pin.DoesNotExist:
                        continue
            # Debug logging - print responses received
            print(f"All Responses: {all_responses_received}")

            return JsonResponse(all_responses_received, safe=False)
        else:
            return Response({
                'status': 'incomplete'
            })


@csrf_exempt
@sync_to_async
@async_to_sync
async def wordSearch(request):
    sentence = request.POST.get("word").strip().lower()
    media_url = os.path.join(BASE_DIR, 'media')
    index = faiss.read_index(f"{media_url}/sentence.index")
    model = SentenceTransformer('all-MiniLM-L6-v2')
    k = 30
    query_vector = model.encode([sentence])

    # Debug logging
    print(f"Query Vector Shape: {query_vector.shape}")

    D, I = index.search(query_vector, k)

    # Debug logging
    print(f"Search Results: {I}")

    all_responses_received = []
    for e in I[0]:
        if e != -1:
            pin = await get_pin(e)
            if pin:
                image = str(pin.image)
                all_responses_received.append({
                    'image': image, 'name': pin.name,
                    'slug': pin.slug
                })
            else:
                # Debug logging
                print(f"Pin with id {e} does not exist.")
                
    # Debug logging
    print(f"All Responses: {all_responses_received}")

    return JsonResponse(all_responses_received, safe=False)

@sync_to_async
def get_pin(id):
    try:
        return Pin.objects.get(id=id)
    except Pin.DoesNotExist:
        # Debug logging
        print(f"Pin with id {id} does not exist in the database.")
        return None

