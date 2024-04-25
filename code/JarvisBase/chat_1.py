import os
import openai
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from elevenlabs import generate
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from streamlit_chat import message
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Constants
TEMP_AUDIO_PATH = "temp_audio.wav"
AUDIO_FORMAT = "audio/wav"
chromadb_collection_name= "ada_hf_hub_doc_collection"

# Load environment variables from .env file and return the keys
os.environ['OPENAI_API_KEY'] = ""
os.environ["ACTIVELOOP_TOKEN"] = ""
os.environ["GOOGLE_API_KEY"]= ""
os.environ["GOOGLE_CSE_ID"]= ""
os.environ["HUGGINGFACEHUB_API_TOKEN"]= ""
os.environ["COHERE_API_KEY"] = ""
os.environ["WOLFRAM_ALPHA_APPID"] = ""
os.environ["SERPAPI_API_KEY"]= ""

# Load embeddings and DeepLake database
# Load embeddings and DeepLake database
def load_embeddings_and_database(collection_name):
    embed_ada =  OpenAIEmbeddings(model="text-embedding-ada-002")
    client = chromadb.Client()
    db = Chroma(client = client, collection_name = collection_name, embedding_function=embed_ada)
#     db = client.get_collection(name=collection_name, embedding_function=embed_ada)
    print("There are", db._collection.count(), "in the collection")
    return db

# Transcribe audio using OpenAI Whisper API
def transcribe_audio(audio_file_path, openai_key):
    openai.api_key = openai_key
    try:
        with open(audio_file_path, "rb") as audio_file:
            response = openai.Audio.transcribe("whisper-1", audio_file)
        return response["text"]
    except Exception as e:
        print(f"Error calling Whisper API: {str(e)}")
        return None

# Record audio using audio_recorder and transcribe using transcribe_audio
def record_and_transcribe_audio():
    audio_bytes = audio_recorder()
    transcription = None
    if audio_bytes:
        st.audio(audio_bytes, format=AUDIO_FORMAT)

        with open(TEMP_AUDIO_PATH, "wb") as f:
            f.write(audio_bytes)

        if st.button("Transcribe"):
            transcription = transcribe_audio(TEMP_AUDIO_PATH, openai.api_key)
            os.remove(TEMP_AUDIO_PATH)
            display_transcription(transcription)

    return transcription

# Display the transcription of the audio on the app
def display_transcription(transcription):
    if transcription:
        st.write(f"Transcription: {transcription}")
        with open("audio_transcription.txt", "w+") as f:
            f.write(transcription)
    else:
        st.write("Error transcribing audio.")

# Get user input from Streamlit text input field
def get_user_input(transcription):
    return st.text_input("", value=transcription if transcription else "", key="input")

# Search the database for a response based on the user's query
# Search the database for a response based on the user's query
def search_db(user_input, db):
    print(user_input)
    retriever = db.as_retriever(search_kwargs={"k": 2})
    model = ChatOpenAI(model='gpt-3.5-turbo')
    qa = RetrievalQA.from_llm(model, retriever=retriever, return_source_documents=True)
    return qa({'query': user_input})

# Display conversation history using Streamlit messages
def display_conversation(history):
    for i in range(len(history["generated"])):
        message(history["past"][i], is_user=True, key=str(i) + "_user")
        message(history["generated"][i],key=str(i))
        #Voice using Eleven API
        voice= "Bella"
        text= history["generated"][i]
        audio = generate(text=text, voice=voice,api_key=eleven_api_key)
        st.audio(audio, format='audio/mp3')

# Main function to run the app
def main():
    # Initialize Streamlit app with a title
    st.write("# JarvisBase 🧙")
   
    # Load embeddings and the DeepLake database
    db = load_embeddings_and_database(chromadb_collection_name)

    # Record and transcribe audio
    transcription = record_and_transcribe_audio()

    # Get user input from text input or audio transcription
    user_input = get_user_input(transcription)

    # Initialize session state for generated responses and past messages
    if "generated" not in st.session_state:
        st.session_state["generated"] = ["I am ready to help you"]
    if "past" not in st.session_state:
        st.session_state["past"] = ["Hey there!"]
        
    # Search the database for a response based on user input and update session state
    if user_input:
        output = search_db(user_input, db)
        print(output['source_documents'])
        st.session_state.past.append(user_input)
        response = str(output["result"])
        st.session_state.generated.append(response)

    # Display conversation history using Streamlit messages
    if st.session_state["generated"]:
        display_conversation(st.session_state)

# Run the main function when the script is executed
if __name__ == "__main__":
    main()