from paperqa import Settings, ask

settings = Settings()
settings.agent.index.name = "lfrqa_science_index_complete"
settings.agent.index.paper_directory = (
    "rag-qa-benchmarking/lfrqa/science_docs_for_paperqa"
)
settings.agent.index.index_directory = (
    "rag-qa-benchmarking/lfrqa/science_docs_for_paperqa_index"
)

settings.agent.index.manifest_file = "manifest.csv"

settings.parsing.use_doc_details = False
settings.parsing.defer_embedding = True
settings.agent.index.concurrency = 30000

answer_response = (
    ask(
        "$5^n+n$ is never prime?",
        settings=settings,
    ),
)

print("_" * 100)
