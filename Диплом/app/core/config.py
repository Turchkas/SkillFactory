from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent
CACHE_DIR = BASE_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Models
EMBEDDING_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
NLI_MODEL_NAME = "cointegrated/rubert-base-cased-nli-threeway"

# Dataset
DATASET_NAME = "MilyaShams/nli_fever-ru_10k"
TRAIN_SPLIT = "train"
TEST_SPLIT = "test"

# Retrieval
TOP_K = 5

# Cache files
CORPUS_SENTENCES_PATH = CACHE_DIR / "corpus_sentences.txt"
FAISS_INDEX_PATH = CACHE_DIR / "faiss.index"