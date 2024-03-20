import os
from langchain_astradb import AstraDBVectorStore
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from datasets import load_dataset
from dotenv import load_dotenv

load_dotenv()

ASTRA_DB_APPLICATION_TOKEN = os.environ.get("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_API_ENDPOINT = os.environ.get("ASTRA_DB_API_ENDPOINT")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")


embedding = OpenAIEmbeddings()
vstore = AstraDBVectorStore(
    embedding=embedding,
    collection_name="test",
    token=os.environ["ASTRA_DB_APPLICATION_TOKEN"],
    api_endpoint=os.environ["ASTRA_DB_API_ENDPOINT"],
)


philo_dataset = load_dataset("datastax/philosopher-quotes")["train"]
print("An example entry:")
print(philo_dataset[16])
print("\n")


docs = []
for entry in philo_dataset:
    metadata = {"author": entry["author"]}
    if entry["tags"]:
        # Add metadata tags to the metadata dictionary
        for tag in entry["tags"].split(";"):
            metadata[tag] = tag
    # Add a LangChain document with the quote and metadata tags
    doc = Document(page_content=entry["quote"], metadata=metadata)
    docs.append(doc)


inserted_ids = vstore.add_documents(docs)
print(f"\nInserted {len(inserted_ids)} documents.")


results = vstore.similarity_search("The Lord is my shepherd", k=3)
print(f"\nResults Type: {type(results)}")
for res in results:
    print(f"* {res.page_content} [{res.metadata}]")
