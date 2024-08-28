



from langchain_community.vectorstores import Chroma

from langchain import hub
from langchain_core.runnables import RunnablePassthrough


from langchain_core.output_parsers import StrOutputParser
from langchain_openai import OpenAIEmbeddings



import getpass
import os
from langchain_groq import ChatGroq

RANDOM_SEED = 224  # Fixed seed for reproducibility
os.environ['GROQ_API_KEY'] 
os.environ["OPENAI_API_KEY"]


# Embed and index
embedding = OpenAIEmbeddings(
    model="text-embedding-3-small",
    # With the `text-embedding-3` class
    # of models, you can specify the size
    # of the embeddings you want returned.
    # dimensions=1024
)


CHROMA_DIR = "feynman_storage"



def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")


llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
        )




def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



def chroma_db_exists():
    """
    Check if the Chroma database exists.

    Returns:
    - bool: True if the database exists, False otherwise.
    """
    return os.path.exists(CHROMA_DIR)


    # Check if Chroma database exists

def get_retriever():
    
    # Load existing Chroma vector store

    vectorstore_feynman = Chroma(

        embedding_function=embedding,
        persist_directory=CHROMA_DIR,
    )
    print("Loaded existing Chroma vector store")

    retriever = vectorstore_feynman.as_retriever()
    return retriever

print("Indexing complete")



def answer_raptor(question: str) -> str:
    retriever = get_retriever()



    prompt = hub.pull("rlm/rag-prompt")

    # Chain
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain.invoke(question)





