"""Project-wide configuration.

Paths, JD blueprint, and tuning knobs. Editable from a single place.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Set


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
MODELS_DIR = ARTIFACTS_DIR / "models"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_CANDIDATES_JSONL = (
    PROJECT_ROOT
    / "[PUB] India_runs_data_and_ai_challenge"
    / "[PUB] India_runs_data_and_ai_challenge"
    / "India_runs_data_and_ai_challenge"
    / "candidates.jsonl"
)
DEFAULT_JOB_DESCRIPTION = (
    PROJECT_ROOT
    / "[PUB] India_runs_data_and_ai_challenge"
    / "[PUB] India_runs_data_and_ai_challenge"
    / "India_runs_data_and_ai_challenge"
    / "job_description.docx"
)
DEFAULT_OUTPUT_CSV = PROJECT_ROOT / "submission.csv"
DEFAULT_VALIDATOR = (
    PROJECT_ROOT
    / "[PUB] India_runs_data_and_ai_challenge"
    / "[PUB] India_runs_data_and_ai_challenge"
    / "India_runs_data_and_ai_challenge"
    / "validate_submission.py"
)

# Artifact filenames
CANDIDATES_PARQUET = ARTIFACTS_DIR / "candidates.parquet"
GRAPH_PICKLE = ARTIFACTS_DIR / "graph.pkl"
TITLE_EMB_NPY = ARTIFACTS_DIR / "title_embeddings.npy"
BM25_PICKLE = ARTIFACTS_DIR / "bm25_index.pkl"
BM25_TOKENS_PICKLE = ARTIFACTS_DIR / "bm25_tokens.pkl"
DENSE_NPY = ARTIFACTS_DIR / "dense_index.npy"
DENSE_IDS_JSON = ARTIFACTS_DIR / "dense_ids.json"
RANKER_TXT = ARTIFACTS_DIR / "ranker.txt"
BLUEPRINT_JSON = ARTIFACTS_DIR / "blueprint.json"

# ---------------------------------------------------------------------------
# Role Blueprint — Senior AI Engineer (Redrob founding team)
# Extracted from job_description.docx. Kept here in code so the sandbox
# can run without parsing the .docx and so the blueprint is auditable.
# ---------------------------------------------------------------------------

CORE_COMPETENCIES: Set[str] = {
    # ML foundations
    "PyTorch", "TensorFlow", "Transformers", "Hugging Face", "HuggingFace",
    "LLM", "LLMs", "Large Language Models", "Fine-tuning", "Fine-Tuning",
    "LoRA", "QLoRA", "PEFT", "Prompt Engineering",
    # Retrieval / ranking infra
    "RAG", "Retrieval Augmented Generation", "Vector Database", "Vector Databases",
    "Pinecone", "Weaviate", "Qdrant", "Milvus", "FAISS", "OpenSearch",
    "Elasticsearch", "BM25", "Hybrid Search", "Dense Retrieval",
    "Sentence Transformers", "sentence-transformers", "BGE", "E5",
    "Embeddings", "Semantic Search",
    # IR / ranking theory
    "Learning to Rank", "Learning-to-Rank", "Learning2Rank", "LTR",
    "XGBoost", "LightGBM", "CatBoost", "NDCG", "MRR", "MAP",
    "LambdaRank", "LambdaMART",
    # Engineering
    "Python", "AWS", "GCP", "Azure", "Docker", "Kubernetes",
    "Distributed Systems", "MLOps", "ML Ops", "Inference",
    # Eval / experimentation
    "A/B Testing", "A/B Test", "Experimentation", "Evaluation",
    "Offline Evaluation", "Online Evaluation", "Eval Framework", "Evaluation Framework",
}

ADJACENT_COMPETENCIES: Set[str] = {
    "Spark", "PySpark", "Airflow", "Kafka", "dbt", "Snowflake",
    "Databricks", "BigQuery", "Redshift", "Beam", "Flink",
    "Feature Store", "Feature Engineering", "MLflow", "Kubeflow",
    "Triton", "TensorRT", "ONNX", "RecSys", "Recommendation",
    "Recommendation Systems", "Search", "Ranking", "Personalization",
    "Open Source", "Open-Source", "GitHub", "NLP", "Computer Vision",
    "LangChain", "LlamaIndex", "LangGraph",
}

NEGATIVE_COMPETENCIES: Set[str] = {
    # these are not disqualifiers by themselves — used to detect
    # "Marketing Manager + AI keywords" profiles during honeypot scoring
    "SEO", "Photoshop", "Illustrator", "Content Writing", "Sales",
    "Marketing", "Accounting", "Mechanical Design", "Civil Engineering",
    "Customer Support", "Operations Management", "HR Management",
    "Business Analysis", "Six Sigma", "SAP", "Recruiting",
    "Brand Design", "Talent Acquisition", "CAD", "SolidWorks",
    "ANSYS", "FEA", "DFM", "DFMA",
}

# Phrases that the structured high-recall gate uses to keep adjacent roles
# like Search/Relevance Engineers in the pool.
STRUCTURED_GATE_TERMS: Set[str] = {
    "pytorch", "tensorflow", "hugging", "transformers", "llm", "rag",
    "retrieval", "recommendation", "ranking", "ranker", "search", "embedding",
    "vector", "faiss", "pinecone", "weaviate", "qdrant", "milvus",
    "opensearch", "elasticsearch", "bm25", "xgboost", "lightgbm", "catboost",
    "learning to rank", "learning-to-rank", "learning2rank", "lambdarank",
    "lora", "qlora", "peft", "fine-tuning", "fine tuning", "fine-tune",
    "sentence-transformer", "sentence transformer", "bge", "e5",
    "ml engineer", "ai engineer", "applied scientist", "applied ml",
    "machine learning", "data scientist", "nlp engineer", "nlp",
    "search engineer", "search relevance", "relevance engineer",
    "recommender", "recommendation systems", "ranker", "rank model",
    "feature store", "feature engineering", "mlops", "kubeflow", "mlflow",
    "triton", "tensorrt", "onnx", "distributed training", "inference",
    "hybrid search", "dense retrieval", "sparse retrieval",
    "promotion velocity", "reranking", "re-ranking", "rerank", "re-rank",
    "vector database", "vector search",
}

TARGET_TITLES: Set[str] = {
    "ML Engineer", "Senior ML Engineer", "Staff ML Engineer", "Principal ML Engineer",
    "Machine Learning Engineer", "Senior Machine Learning Engineer",
    "AI Engineer", "Senior AI Engineer", "Staff AI Engineer",
    "Applied Scientist", "Senior Applied Scientist", "Staff Applied Scientist",
    "Applied ML Engineer", "Senior Applied ML Engineer",
    "Research Engineer", "Senior Research Engineer",
    "Recommendation Systems Engineer", "Search Engineer", "Search Relevance Engineer",
    "Relevance Engineer", "Ranking Engineer", "Senior Ranking Engineer",
    "NLP Engineer", "Senior NLP Engineer",
    "Data Scientist", "Senior Data Scientist", "Staff Data Scientist",
    "Machine Learning Scientist", "Senior Machine Learning Scientist",
    "AI Specialist", "Senior AI Specialist",
    "Founding Engineer", "Founding ML Engineer",
}

# Title similarity groups — used for the graduated title fit feature.
# Each group has a target role weight; matches accumulate.
TITLE_ROLE_GROUPS: Dict[str, List[str]] = {
    "applied_ml": [
        "ml engineer", "machine learning engineer", "ai engineer",
        "applied scientist", "applied ml", "research engineer",
        "machine learning scientist", "ai specialist",
    ],
    "retrieval_ranking": [
        "search engineer", "search relevance engineer", "relevance engineer",
        "ranking engineer", "recommendation systems engineer",
        "recommender", "recommendation systems",
    ],
    "nlp_llm": [
        "nlp engineer", "nlp", "llm engineer", "generative ai engineer",
    ],
    "data_science": [
        "data scientist", "senior data scientist", "applied data scientist",
    ],
    "data_platform": [
        "data engineer", "senior data engineer", "analytics engineer",
        "ml platform engineer", "ml infrastructure engineer", "mlops engineer",
    ],
    "generic_swe": [
        "software engineer", "senior software engineer", "backend engineer",
        "full stack developer", "full stack engineer", "platform engineer",
    ],
    "non_target": [
        "marketing manager", "hr manager", "accountant", "sales executive",
        "business analyst", "operations manager", "project manager",
        "graphic designer", "content writer", "civil engineer",
        "mechanical engineer", "customer support", "qa engineer",
        "frontend engineer", "frontend developer", "java developer",
        ".net developer", "mobile developer", "devops engineer",
        "cloud engineer", "data analyst", "frontend",
    ],
}

# Title weights used by the graduated title fit scorer (target role weight).
TITLE_GROUP_WEIGHT: Dict[str, float] = {
    "applied_ml": 1.00,
    "retrieval_ranking": 0.95,
    "nlp_llm": 0.85,
    "data_science": 0.70,
    "data_platform": 0.55,
    "generic_swe": 0.40,
    "non_target": 0.05,
}

# Seniority band (years of experience).
YOE_MIN = 4.0
YOE_IDEAL_LOW = 5.0
YOE_IDEAL_HIGH = 9.0
YOE_MAX_USEFUL = 15.0

# Notice period & location
NOTICE_OK_DAYS = 30
NOTICE_HARD_DAYS = 90
PREFERRED_LOCATIONS = {
    "Noida", "Pune", "Hyderabad", "Mumbai", "Bangalore", "Bengaluru",
    "Delhi", "Gurgaon", "Gurugram", "Delhi NCR",
}
COUNTRY_OK = {"India"}

# Companies whose culture the JD flags ("title-chasers", "framework enthusiasts",
# pure-services). Not strict exclusions but used as a soft penalty in features.
IT_SERVICES_PURE_PLAY = {
    "TCS", "Infosys", "Wipro", "HCL", "Tech Mahindra", "Cognizant",
    "Capgemini", "Accenture", "Mindtree", "L&T Infotech", "Mphasis",
    "Persistent", "Zensar", "Hexaware", "Birlasoft", "Cyient",
}
PRODUCT_COMPANY_HINTS = {
    "Razorpay", "Zerodha", "CRED", "PhonePe", "Paytm", "Swiggy",
    "Zomato", "Ola", "Flipkart", "Meesho", "Urban Company",
    "Dream11", "Groww", "Cars24", "MPL", "Lenskart", "Nykaa",
    "Postman", "Freshworks", "Zoho", "Chargebee", "Razorpay",
    "BrowserStack", "Sprinklr", "CleverTap", "MoEngage", "ShareChat",
    "InMobi", "Oyo", "OYO", "Rivigo", "Delhivery", "Udaan",
    "Microsoft", "Google", "Meta", "Amazon", "Apple", "Netflix",
    "Stripe", "Airbnb", "Uber", "Linkedin", "LinkedIn",
    "Salesforce", "Adobe", "Nvidia", "NVIDIA", "Intel", "IBM",
    "Oracle", "SAP", "ServiceNow", "Snowflake", "Databricks",
    "Redrob", "Redrob AI", "RedrobAI",
    "OpenAI", "Anthropic", "Cohere", "Hugging Face", "HuggingFace",
    "Pinecone", "Weaviate", "Qdrant",
}

# ---------------------------------------------------------------------------
# Retrieval knobs
# ---------------------------------------------------------------------------

BM25_TOP_K = 3000
DENSE_TOP_K = 3000
STRUCTURED_GATE_TOP_K = 8000   # recall-friendly; not a hard filter
RRF_K = 60
SHORTLIST_N = 4000

# DENSE_MODEL can be overridden via env var REDROB_DENSE_MODEL
# Default uses MiniLM for fast offline runs; stronger BGE models are
# supported by setting the env var before running.
DEFAULT_DENSE_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
# Stronger alternatives (require the model to be cached locally):
#   "BAAI/bge-small-en-v1.5"
#   "BAAI/bge-base-en-v1.5"
#   "BAAI/bge-large-en-v1.5"
DENSE_MODEL_OPTIONS = [
    DEFAULT_DENSE_MODEL,
    "BAAI/bge-small-en-v1.5",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
]

# BM25 / dense query
JD_QUERY_TERMS = [
    "Senior AI Engineer",
    "Applied ML Engineer",
    "retrieval augmented generation",
    "vector search",
    "hybrid search",
    "BM25 FAISS Pinecone",
    "PyTorch transformers",
    "fine-tuning LoRA QLoRA PEFT",
    "learning to rank XGBoost LightGBM",
    "NDCG MRR MAP",
    "embeddings sentence-transformers BGE E5",
    "vector database Milvus Qdrant Weaviate",
    "production machine learning systems",
    "evaluation framework A/B testing",
    "ranker ranking recommendation",
    "MLOps inference distributed",
]

# ---------------------------------------------------------------------------
# Ranker knobs
# ---------------------------------------------------------------------------

RANKER_PARAMS = dict(
    objective="lambdarank",
    metric="ndcg",
    eval_at=[10, 50],
    num_leaves=63,
    learning_rate=0.05,
    n_estimators=600,
    lambdarank_truncation_level=50,
    min_data_in_leaf=20,
    feature_fraction=0.85,
    bagging_fraction=0.85,
    bagging_freq=5,
    random_state=42,
    verbose=-1,
)

# Hard exclude honeypots at/above this penalty.
HONEYPOT_HARD_EXCLUDE = 0.5

# Reasoning template rotation
REASONING_TEMPLATE_COUNT = 15

# ---------------------------------------------------------------------------
# JD-criticality weighting (for skill feature engineering)
# ---------------------------------------------------------------------------

JD_CRITICAL: set[str] = {
    # The JD's "absolutely need" — weight 2.0
    "RAG", "Retrieval Augmented Generation",
    "FAISS", "Pinecone", "Weaviate", "Qdrant", "Milvus",
    "OpenSearch", "Elasticsearch",
    "BM25", "Hybrid Search", "Dense Retrieval",
    "Sentence Transformers", "BGE", "E5", "Embeddings",
    "Learning to Rank", "LambdaRank", "NDCG", "MRR", "MAP",
    "XGBoost", "LightGBM", "CatBoost",
    "PyTorch", "Transformers", "LLM", "Fine-tuning", "LoRA", "QLoRA", "PEFT",
    "Prompt Engineering", "Hugging Face", "HuggingFace",
    "Vector Database", "Vector Databases",
}
JD_NICE_TO_HAVE: set[str] = {
    # JD "nice to have" — weight 1.0
    "Kubernetes", "Docker", "AWS", "GCP", "Azure", "Python",
    "MLOps", "ML Ops", "MLflow", "Kubeflow",
    "Triton", "TensorRT", "ONNX",
    "Distributed Systems", "Inference", "Evaluation",
    "A/B Testing", "A/B Test", "Experimentation",
    "NLP", "Computer Vision",
    "Recommendation", "Recommendation Systems", "Search", "Ranking",
    "RecSys", "LlamaIndex", "LangChain", "LangGraph",
    "Open Source", "Open-Source",
}

# ---------------------------------------------------------------------------
# Multi-axis PPR: 5 JD axes
# ---------------------------------------------------------------------------

JD_AXES: dict[str, list[str]] = {
    "applied_ml": [
        "PyTorch", "TensorFlow", "Transformers", "Hugging", "LLM", "LoRA",
        "QLoRA", "PEFT", "Fine-tuning", "Fine-Tuning", "Prompt Engineering",
        "NLP", "scikit-learn",
    ],
    "retrieval_rank": [
        "FAISS", "Pinecone", "Weaviate", "Qdrant", "Milvus", "OpenSearch",
        "Elasticsearch", "BM25", "Hybrid Search", "Dense Retrieval",
        "Learning to Rank", "LambdaRank", "NDCG", "MRR", "MAP",
        "XGBoost", "LightGBM", "CatBoost",
        "Sentence Transformers", "BGE", "E5", "Embeddings",
    ],
    "nlp_llm": [
        "LLM", "Fine-tuning", "LoRA", "QLoRA", "PEFT", "Prompt Engineering",
        "RAG", "Retrieval Augmented Generation", "Vector Database",
        "LangChain", "LlamaIndex", "LangGraph", "NLP", "Transformers",
    ],
    "production_eng": [
        "Kubernetes", "Docker", "AWS", "GCP", "Azure", "Distributed Systems",
        "MLOps", "MLflow", "Kubeflow", "Triton", "TensorRT", "ONNX",
        "Inference", "Evaluation",
    ],
    "product_company": [
        "Razorpay", "Zerodha", "CRED", "PhonePe", "Paytm", "Swiggy", "Zomato",
        "Ola", "Flipkart", "Meesho", "Urban Company", "Dream11", "Groww",
        "Cars24", "Postman", "Freshworks", "Zoho", "Chargebee", "BrowserStack",
        "Sprinklr", "CleverTap", "MoEngage", "ShareChat", "InMobi",
        "Microsoft", "Google", "Meta", "Amazon", "Apple", "Netflix",
        "Stripe", "Airbnb", "Uber", "LinkedIn", "Salesforce", "Adobe",
        "NVIDIA", "Snowflake", "Databricks",
        "OpenAI", "Anthropic", "Cohere", "Hugging Face",
        "Pinecone", "Weaviate", "Qdrant",
    ],
}

# ---------------------------------------------------------------------------
# Company tier (for company-quality feature)
# ---------------------------------------------------------------------------

COMPANY_TIER: dict[int, set[str]] = {
    3: {
        "Google", "Meta", "Microsoft", "Amazon", "Apple", "Netflix",
        "Stripe", "Airbnb", "Uber", "LinkedIn",
        "Salesforce", "Adobe", "NVIDIA", "Snowflake", "Databricks",
        "OpenAI", "Anthropic", "Cohere", "Hugging Face",
        "Pinecone", "Weaviate", "Qdrant",
    },
    2: {
        "Razorpay", "Zerodha", "CRED", "PhonePe", "Paytm",
        "Swiggy", "Zomato", "Flipkart", "Meesho", "Urban Company",
        "Postman", "Freshworks", "Zoho", "Chargebee",
        "BrowserStack", "Sprinklr", "CleverTap", "MoEngage", "ShareChat",
        "InMobi", "Dream11", "Groww", "Cars24",
        "Ola", "Rivigo", "Delhivery", "Udaan", "Oyo",
        "Intel", "IBM", "Oracle", "Redrob",
    },
}
