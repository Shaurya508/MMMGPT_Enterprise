import io
import os
import re
import time
import sys
from pptx import Presentation
from PIL import Image
import numpy as np
import easyocr
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
import config
from PIL import UnidentifiedImageError
import subprocess

# Set environment variables
for key, value in config.ENV_VARS.items():
    os.environ[key] = value

def extract_text_from_image(shape, reader, slide_number, image_number):
    """Extract text from images in PPT slides"""
    try:
        image_bytes = shape.image.blob
        image_stream = io.BytesIO(image_bytes)
        img = Image.open(image_stream)

        image_path = os.path.join(config.EXTRACTED_IMG_DIR, f"slide_{slide_number}image{image_number}.png")
        
        # Check if image is in WMF format and handle accordingly
        if img.format == 'WMF':
            # Save the WMF image temporarily
            temp_wmf_path = os.path.join(config.EXTRACTED_IMG_DIR, f"slide_{slide_number}image{image_number}.wmf")
            with open(temp_wmf_path, "wb") as f:
                f.write(image_bytes)
            
            # Convert WMF to PNG using ImageMagick
            converted_image_path = image_path
            # convert_command = f"convert {temp_wmf_path} {converted_image_path}"
            convert_command = f"convert '{temp_wmf_path}' '{converted_image_path}'"

            subprocess.run(convert_command, shell=True, check=True)

        else:
            # Save image directly as PNG
            img.save(image_path)
        
        # Proceed with OCR processing
        img_np = np.array(img.convert('L'))
        result = reader.readtext(img_np, detail=1)

        rows = []
        current_row = []
        previous_y = None
        tolerance = 15

        for entry in result:
            bbox, text, confidence = entry
            x_min, y_min = bbox[0]

            if previous_y is None or abs(y_min - previous_y) <= tolerance:
                current_row.append(text)
            else:
                rows.append(current_row)
                current_row = [text]
            previous_y = y_min

        if current_row:
            rows.append(current_row)

        table_str = ""
        for row in rows:
            table_str += '\t'.join(row) + '\n'

        image_opening_flag = f"<|S{slide_number}I{image_number}|>"
        image_closing_flag = f"</|S{slide_number}I{image_number}|>"
        return f"{image_opening_flag}\n{table_str}\n{image_closing_flag}"
    
    except UnidentifiedImageError:
        print(f"Skipping unsupported image format on slide {slide_number}, image {image_number}.")
        return ""

def extract_title(shape, slide_number, chart_count):
    """Extract chart title and save chart image from the slide"""
    chart_title = ""
    if shape.chart.has_title:
        chart_title = shape.chart.chart_title.text_frame.text

    title_opening_flag = f"<|S{slide_number}C{chart_count}|>"
    title_closing_flag = f"</|S{slide_number}C{chart_count}|>"
    return f"{title_opening_flag}\n{chart_title}\n{title_closing_flag}"

def get_ppt_data(file_name):
    """Extract text, tables, and images from PPT"""
    prs = Presentation(file_name)
    extracted_text = ""
    reader = easyocr.Reader(['en'])

    for slide_number, slide in enumerate(prs.slides, start=1):
        image_count = 0
        chart_count = 0
        for shape in slide.shapes:
            if shape.has_table:
                table = shape.table
                for row in table.rows:
                    row_data = [cell.text for cell in row.cells]
                    extracted_text += str(row_data) + '\n'

            if hasattr(shape, "text"):
                text = shape.text.replace('\v', '\n').replace('\x0b', '\n')
                extracted_text += text + '\n'

            if hasattr(shape, "image"):
                image_count += 1
                extracted_text += extract_text_from_image(shape, reader, slide_number, image_count) + '\n'

            if shape.has_chart:
                chart_count += 1
                extracted_text += extract_title(shape, slide_number, chart_count) + '\n'

    return extracted_text

def get_text_chunks_with_flags(content):
    """Chunk text with metadata flags"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, 
        chunk_overlap=config.CHUNK_OVERLAP
    )
    chunks_with_metadata = []
    current_metadata = set()

    image_flag_pattern = r"<\|S(\d+)I(\d+)\|>"
    chart_flag_pattern = r"<\|S(\d+)C(\d+)\|>"
    image_closing_pattern = r"</\|S(\d+)I(\d+)\|>"
    chart_closing_pattern = r"</\|S(\d+)C(\d+)\|>"

    text_chunks = text_splitter.split_text(content)
    
    for chunk in text_chunks:
        chunk_metadata = set()

        # Process flags and maintain metadata
        for pattern, closing_pattern, prefix in [
            (image_flag_pattern, image_closing_pattern, "slide_%s_image_%s.png"),
            (chart_flag_pattern, chart_closing_pattern, "slide_%s_chart_%s.png")
        ]:
            opening_flags = re.findall(pattern, chunk)
            for match in opening_flags:
                file_path = prefix % match
                chunk_metadata.add(file_path)
                current_metadata.add(file_path)

            closing_flags = re.findall(closing_pattern, chunk)
            for match in closing_flags:
                file_path = prefix % match
                chunk_metadata.add(file_path)
                current_metadata.discard(file_path)

        chunk_metadata.update(current_metadata)

        # Clean chunk text
        chunk_clean = re.sub(image_flag_pattern, '', chunk)
        chunk_clean = re.sub(image_closing_pattern, '', chunk_clean)
        chunk_clean = re.sub(chart_flag_pattern, '', chunk_clean)
        chunk_clean = re.sub(chart_closing_pattern, '', chunk_clean)

        chunks_with_metadata.append({
            "text": chunk_clean.strip(),
            "metadata": {"image_paths": list(chunk_metadata)}
        })

    return chunks_with_metadata

def main():
    start_time = time.time()
    
    if len(sys.argv) != 2:
        print("Usage: python preprocess.py <ppt_file_path>")
        sys.exit(1)

    ppt_file_name = sys.argv[1]
    
    # Initialize embedding model
    embeddings = OllamaEmbeddings(model=config.EMBEDDING_MODEL)
    
    # Extract and process PPT data
    extracted_text = get_ppt_data(ppt_file_name)
    chunks_with_metadata = get_text_chunks_with_flags(extracted_text)
    
    # Create FAISS database
    texts = [item['text'] for item in chunks_with_metadata]
    metadatas = [item['metadata'] for item in chunks_with_metadata]
    vector_store = FAISS.from_texts(texts, embeddings, metadatas=metadatas)
    vector_store.save_local(config.DB_NAME)
    
    print(f"FAISS database created and saved as '{config.DB_NAME}'")
    print(f"Overall Time Taken = {time.time() - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
