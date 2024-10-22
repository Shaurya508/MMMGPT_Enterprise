import os

# Database configuration
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100
VERSION_NUMBER = 1
DB_NAME = f"PPT_DB_nomic_chunk_size-{CHUNK_SIZE}__overlap-{CHUNK_OVERLAP}_v{VERSION_NUMBER}"

# Model configuration
EMBEDDING_MODEL = "zxf945/nomic-embed-text:latest"
CHAT_MODEL = "llama3.2:3b"

# Directory configuration
MAIN_IMG_DIR = "Images"
FULL_SLIDE_IMG_DIR = os.path.join(MAIN_IMG_DIR, "Full Slide Images")
EXTRACTED_IMG_DIR = os.path.join(MAIN_IMG_DIR, "Extracted Images")
EXTRACTED_CHART_DIR = os.path.join(MAIN_IMG_DIR, "Extracted Charts")
PPT_DIRECTORY = "uploaded_ppt"

# Environment variables
ENV_VARS = {
    "GRPC_VERBOSITY": "ERROR",
    "GRPC_TRACE": "",
    "TF_CPP_MIN_LOG_LEVEL": "3",
    "HF_HOME": "F:/D Drive Backup/HuggingFaceModelsCache",
    "OLLAMA_MODELS": "F:/D Drive Backup/OllamaModels"
}