import glob
import json
import os
import pandas as pd
import ollama

# Folder containing the images extracted from your zip file
IMAGE_FOLDER = "soil_images"
MODEL_NAME = "llava"  # Uses standard LLaVA vision architecture supported by all Ollama servers

# Prompt instructing the model to extract specific fields in JSON format
PROMPT = """
Analyze this soil test report / image and extract the following parameters:
- Latitude
- Longitude
- Village
- N (Nitrogen value, specify units if present)
- P (Phosphorus value, specify units if present)
- K (Potassium value, specify units if present)
- OC (Organic Carbon value, specify units or % if present)

Return ONLY a valid JSON object with these exact keys:
{
  "Latitude": "extracted value or N/A",
  "Longitude": "extracted value or N/A",
  "Village": "extracted value or N/A",
  "N": "extracted value or N/A",
  "P": "extracted value or N/A",
  "K": "extracted value or N/A",
  "OC": "extracted value or N/A"
}
Do not include any extra commentary or markdown outside the JSON object.
"""

extracted_data = []

# Gather all image files
image_paths = sorted(
    glob.glob(os.path.join(IMAGE_FOLDER, "*.png")) +
    glob.glob(os.path.join(IMAGE_FOLDER, "*.jpg")) +
    glob.glob(os.path.join(IMAGE_FOLDER, "*.jpeg"))
)

print(f"Found {len(image_paths)} images to process using Ollama ({MODEL_NAME})...\n")

for idx, img_path in enumerate(image_paths, start=1):
    filename = os.path.basename(img_path)
    print(f"[{idx}/{len(image_paths)}] Processing: {filename}...")

    try:
        # Call Ollama local vision model
        response = ollama.chat(
            model=MODEL_NAME,
            messages=[{
                'role': 'user',
                'content': PROMPT,
                'images': [img_path]
            }]
        )

        response_text = response['message']['content'].strip()

        # Clean JSON markdown if wrapped in ```json ... ```
        if response_text.startswith("```"):
            response_text = response_text.split("```")[1]
            if response_text.startswith("json"):
                response_text = response_text[4:].strip()

        # Parse JSON output
        parsed_json = json.loads(response_text)
        parsed_json["Image_Name"] = filename
        extracted_data.append(parsed_json)

    except Exception as e:
        print(f"  --> Error processing {filename}: {e}")
        extracted_data.append({
            "Image_Name": filename,
            "Latitude": "Error",
            "Longitude": "Error",
            "Village": "Error",
            "N": "Error",
            "P": "Error",
            "K": "Error",
            "OC": "Error"
        })

# Convert extracted records to a DataFrame and export to Excel
df = pd.DataFrame(extracted_data)

# Reorder columns nicely
columns_order = ["Image_Name", "Latitude", "Longitude", "Village", "N", "P", "K", "OC"]
df = df.reindex(columns=columns_order)

output_excel = "Soil_Test_Results_Ollama.xlsx"
df.to_excel(output_excel, index=False)
print(f"\nProcessing complete! Data successfully saved to '{output_excel}'.")