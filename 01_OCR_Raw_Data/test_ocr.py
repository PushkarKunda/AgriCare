import easyocr
import json

reader = easyocr.Reader(['en'], gpu=True)

for img_name in ['Screenshot 2026-08-05 125800.png', 'Screenshot 2026-08-05 140120.png']:
    print(f"=== {img_name} ===")
    results = reader.readtext(f"./soil_images/{img_name}", detail=1)
    for bbox, text, prob in results:
        print(f"{text} | bbox: {bbox}")
