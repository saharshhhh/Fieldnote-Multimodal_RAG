# ingest/encoders_check.py
from ingest.encoders import embed_image_query, embed_text_query, embed_texts

a = embed_text_query("why does the unit trip on high pressure")
b = embed_image_query("pressure curve chart")
print(len(a), len(b))            # expect 384 768

d = embed_texts(["High pressure cutout occurs at 650 psig.", "Replace the air filter monthly."])
sim = [sum(x * y for x, y in zip(a, v)) for v in d]
print(sim)                        # first (pressure) should score higher than the second