from paperqa import Settings, ask


settings = Settings()
settings.agent.index.paper_directory = f"/Users/joaquinpolonuer/Documents/rag-qa-arena/lfrqa/science_docs_for_paperqa"
settings.agent.index.index_directory = f"/Users/joaquinpolonuer/Documents/rag-qa-arena/lfrqa/science_docs_for_paperqa_index"

settings.agent.index.manifest_file = "manifest.csv"
# settings.agent.search_count = 15

settings.parsing.use_doc_details = False
settings.parsing.defer_embedding = True
settings.agent.index.concurrency = 1000

# settings.prompts.qa.replace('If the context provides insufficient information reply "I cannot answer."', "")

answer_response = (
    ask(
        "$5^n+n$ is never prime?",
        settings=settings,
    ),
)

print("_" * 100)
