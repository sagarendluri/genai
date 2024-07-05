import os
import streamlit as st
from beyondllm import source, retrieve, embeddings, llms, generator
from beyondllm.embeddings import AzureAIEmbeddings
from beyondllm.llms import AzureOpenAIModel
from beyondllm import source
import secrets

uploaded_data_files = st.file_uploader("Upload files", type="pdf", accept_multiple_files=True, label_visibility="visible")

def uploaded_files(uploaded_data_files):
    if uploaded_data_files is not None:
        print("uploaded_data_files",uploaded_data_files)
        save_path = "./Doctor_prescription_files"
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        filenames = []
        for file in uploaded_data_files:
            file_path = os.path.join(save_path, file.name)
            filenames.append(file_path)
            with open(file_path, "wb") as f:
                f.write(file.getbuffer())
        data = source.fit(filenames, dtype="pdf", chunk_size=1024, chunk_overlap=0)
        return data
        
data = uploaded_files(uploaded_data_files)



endpoint_url = st.secrets.azure_embeddings_credentials.ENDPOINT_URL
azure_key = st.secrets.azure_embeddings_credentials.AZURE_KEY
api_version = st.secrets.azure_embeddings_credentials.API_VERSION
deployment_name = st.secrets.azure_embeddings_credentials.DEPLOYMENT_NAME
BASE_URL = st.secrets.azure_embeddings_credentials.BASE_URL
# DEPLOYMENT_NAME = st.secrets.azure_embeddings_credentials.DEPLOYMENT_NAME
API_KEY = st.secrets.azure_embeddings_credentials.API_KEY

embed_model = AzureAIEmbeddings(
    endpoint_url = endpoint_url,
    azure_key = azure_key,
    api_version= api_version,
    deployment_name=deployment_name
)
data = source.fit(path=pdfs, dtype="pdf",chunk_size=512,chunk_overlap=20)
# text_embedding = embed_model.embed_text(str(data))
retriever = retrieve.auto_retriever(data,embed_model=embed_model,type="normal",top_k=4)
llm = AzureOpenAIModel(model="gpt4",azure_key = API_KEY,deployment_name="gpt-4-32k" ,endpoint_url=BASE_URL,model_kwargs={"max_tokens":512,"temperature":0.1})
# option = st.selectbox( 'Please Select the Patient name?', ('Bobby Jackson', 'Leslie Terry','Danny Smith'))
question = st.text_input("Enter your question")
# question = "what is the Bobby Jackson condition?"

system_prompt = "You are acting like a chat...."

submit=st.button("Get the data")
if submit:
    print(question)
    pipeline = generator.Generate(question=question, retriever=retriever,system_prompt=system_prompt, llm=llm)
    response = pipeline.call()
    st.write(response)
    

