# Default lexical metrics — always available, no extra dependencies.
FITNESS_METRICS = ["f1", "bigram_f1", "bag_of_words", "length_ratio"]

# Optional semantic metrics — require extra packages or API access.
# "bert_score" requires: pip install bert-score
# "llm_judge"  requires: --judge-api-key (or OPENAI_API_KEY env var)
OPTIONAL_METRICS = ["bert_score", "llm_judge"]

# All valid metric names for the --metrics CLI flag
ALL_METRICS = FITNESS_METRICS + OPTIONAL_METRICS
