import gradio as gr
from gradio_pdf import PDF
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.vectorstores import Qdrant
from ctransformers import AutoModelForCausalLM

# Load the embedding model
encoder = SentenceTransformer('jinaai/jina-embedding-b-en-v1')
print("Embedding model loaded...")

# Load the LLM
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Llama-2-7B-Chat-GGUF",
    model_file="llama-2-7b-chat.Q3_K_S.gguf",
    model_type="llama",
    temperature=0.2,
    repetition_penalty=1.5,
    max_new_tokens=300,
)
print("LLM loaded...")

client = QdrantClient(path="./db")

def setup_database(files):
    all_chunks = []
    for file in files:
        pdf_path = file
        reader = PdfReader(pdf_path)
        text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=250, chunk_overlap=50, length_function=len)
        chunks = text_splitter.split_text(text)
        all_chunks.extend(chunks)
    
    print(f"Total chunks: {len(all_chunks)}")
    
    client.recreate_collection(
        collection_name="my_facts",
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE,
        ),
    )
    
    print("Collection created...")

    for idx, chunk in enumerate(all_chunks):
        client.upload_record(
            collection_name="my_facts",
            record=models.Record(
                id=idx,
                vector=encoder.encode(chunk).tolist(),
                payload={"text": chunk}
            )
        )
    
    print("Records uploaded...")

def answer(question):
    hits = client.search(
        collection_name="my_facts",
        query_vector=encoder.encode(question).tolist(),
        limit=3
    )
    
    context = " ".join(hit.payload["text"] for hit in hits)
    system_prompt = "You are a helpful co-worker. Use the provided context to answer user questions. Do not use any other information."
    prompt = f"Context: {context}\nUser: {question}\n{system_prompt}"
    response = llm(prompt)
    return response

def chat(messages):
    if not messages:
        return "Please upload PDF documents to initialize the database."
    last_message = messages[-1]
    return answer(last_message["message"])

screen = gr.Interface(
    fn=chat,
    inputs=gr.Chatbot(allow_flagging="never", placeholder="Type your question here..."),
    outputs="chatbot",
    title="Q&A with PDFs 👩🏻‍💻📓✍🏻💡",
    description="This app facilitates a conversation with PDFs uploaded💡",
    theme="soft",
    live=True,
    allow_screenshot=False,
    allow_flagging=False,
)

# Add a way to upload and setup the database before starting the chat
screen.launch()
