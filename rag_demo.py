from llama_index import SimpleDirectoryReader, VectorStoreIndex, ServiceContext
from llama_index.llms import Ollama

llm = Ollama(model="mistral")
docs = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(docs, service_context=ServiceContext.from_defaults(llm=llm))

query_engine = index.as_query_engine()
while True:
    prompt = input("Ask: ")
    print(query_engine.query(prompt))
