import requests
import pandas as pd
import os
import json
from huggingface_hub import InferenceClient
from dotenv import load_dotenv

load_dotenv()       


HF_TOKEN = os.getenv("HF_TOKEN")
folder = 'transcriptions'

chunks_With_embeddings = []

def create_embedding(text_list):
    # https://github.com/ollama/ollama/blob/main/docs/api.md#generate-embeddings
    r = requests.post("http://localhost:11434/api/embed", json={
        "model": "bge-m3",
        "input": text_list
    })
    
    return r.json()["embeddings"]    
    # print(r.status_code)
    # print(r.json())
    # embedding = r.json()["embeddings"] 
    # return embedding





count = 0
# print(embeddings)
for json_file in os.listdir(folder):
    print(f" Loading Content of the file {json_file}")
    with open(f"{folder}/{json_file}" , 'r') as f:
        data = json.load(f)
    
    
    print(f"Creating embeddings for the file {json_file}")
    for chunk in data:
        text = chunk['text']
        embeddings = create_embedding([text])
        chunks_With_embeddings.append({
            'chunk_id': chunk['chunk_id'],
            'start_time': chunk['start'],
            'end_time': chunk['end'],
            "text": text,
            "embedding": embeddings
        })
        


data_frame = pd.DataFrame.from_records(chunks_With_embeddings)
print(data_frame)
data_frame.to_csv("embeddings.csv" , index = False)
print(f" Total Embeddings Created {count}")

print(" Embeddings With Chunks Saved as a CSV file ")



