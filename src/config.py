# Configuration constants for the tag prediction project

# The 8 focus tags for the prediction task
FOCUS_TAGS = [
    'math',
    'graphs',
    'strings',
    'number theory',
    'trees',
    'geometry',
    'games',
    'probabilities'
]


# Model configurations
# Use HuggingFace Hub model (default) 
CODEBERT_MODEL_NAME = "Ahmedjr/codebert-algorithm-tagger"   
CODEBERT_MAX_LENGTH = 512
CODEBERT_DEFAULT_THRESHOLD = 0.5
CODEBERT_BATCH_SIZE = 32

# Retrieval model configurations
RETRIEVAL_EMBED_MODEL = "BAAI/bge-m3"
RETRIEVAL_INDEX_NAME = "Test"
RETRIEVAL_DEFAULT_K = 3
RETRIEVAL_DEFAULT_MIN_VOTES = 1

# LLM configurations
LLM_MODEL_NAME = "llama-3.1-8b-instant"
LLM_MAX_RETRIES = 3

# Weaviate configuration 
WEAVIATE_CLUSTER_URL = "1scymfxircmnej7ffsnba.c0.europe-west3.gcp.weaviate.cloud"

