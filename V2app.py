import gradio as gr
from gradio_pdf import PDF
from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

from langchain.vectorstores import Qdrant
from qdrant_client.http import models
from ctransformers import AutoModelForCausalLM

# Loading the embedding model
encoder = SentenceTransformer('jinaai/jina-embedding-b-en-v1')
print("Embedding model loaded...")

# Loading the LLM
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

def chat(files, question):
    def get_chunks(text):
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=250,
            chunk_overlap=50,
            length_function=len,
        )
        chunks = text_splitter.split_text(text)
        return chunks

    all_chunks = []
    
    for file in files:
        pdf_path = file
        reader = PdfReader(pdf_path)
        text = ""
        num_of_pages = len(reader.pages)

        for page in range(num_of_pages):
            current_page = reader.pages[page]
            text += current_page.extract_text()

        chunks = get_chunks(text)
        all_chunks.extend(chunks)
    
    print(f"Total chunks: {len(all_chunks)}")
    print("Chunks are ready...")

    client = QdrantClient(path="./db")
    print("DB created...")

    client.recreate_collection(
        collection_name="my_facts",
        vectors_config=models.VectorParams(
            size=encoder.get_sentence_embedding_dimension(),
            distance=models.Distance.COSINE,
        ),
    )

    print("Collection created...")

    li = list(range(len(all_chunks)))
    dic = dict(zip(li, all_chunks))

    client.upload_records(
        collection_name="my_facts",
        records=[
            models.Record(
                id=idx,
                vector=encoder.encode(dic[idx]).tolist(),
                payload={f"chunk_{idx}": dic[idx]}
            ) for idx in dic.keys()
        ],
    )

    print("Records uploaded...")

    hits = client.search(
        collection_name="my_facts",
        query_vector=encoder.encode(question).tolist(),
        limit=3
    )
    context = []
    for hit in hits:
        context.append(list(hit.payload.values())[0])
    
    context = " ".join(context)

    system_prompt = """You are a helpful co-worker, you will use the provided context to answer user questions.
    Read the given context before answering questions and think step by step. If you cannot answer a user question based on 
    the provided context, inform the user. Do not use any other information for answering user. Provide a detailed answer to the question."""

    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    SYSTEM_PROMPT = B_SYS + system_prompt + E_SYS

    instruction = f""" 
    Context: {context}
    User: {question}"""

    prompt_template = B_INST + SYSTEM_PROMPT + instruction + E_INST
    print(prompt_template)
    result = llm(prompt_template)
    return result

screen = gr.Interface(
    fn=chat,
    inputs=[gr.File(label="Upload PDFs", file_count="multiple"), gr.Textbox(lines=10, placeholder="Enter your question here 👉")],
    outputs=gr.Textbox(lines=10, placeholder="Your answer will be here soon 🚀"),
    title="Q&A with PDFs 👩🏻‍💻📓✍🏻💡",
    description="This app facilitates a conversation with PDFs uploaded💡",
    theme="soft",
)

screen.launch()
