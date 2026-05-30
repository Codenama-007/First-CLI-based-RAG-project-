import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import requests
import ast
import numpy as np
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


# Initializing The Ollama Model
llm = ChatOllama(model="gemma3:1b" , temperature=0.9)





def create_embedding(text_list):

    r = requests.post(
        "http://localhost:11434/api/embed",
        json={
            "model": "bge-m3",
            "input": text_list
        }
    )

    return r.json()["embeddings"]


# Load CSV
data = pd.read_csv("embeddings.csv")
data["embedding"] = data["embedding"].apply(ast.literal_eval)


# Flatten [[vector]] -> [vector]
data["embedding"] = data["embedding"].apply(lambda x: x[0])


# first_five_rows = data.head(5)


# print(first_five_rows['embedding'].shape)
# print(np.vstack(first_five_rows['embedding'].values))
# print(np.vstack(first_five_rows['embedding']).shape)

while True:
    question = input(" Ask a Question (Type Exit or q to quit) :) ")

    if question.lower() in ['exit', 'q']:
        print("Goodbye! Have a great day!")
        break

    question_to_embedding = create_embedding([question])[0]
    

    similarities = cosine_similarity(np.vstack(data['embedding']) , [question_to_embedding]).flatten()
    # print(similarities)
    # print(similarities.argsort()[-3::-1])
    max_index = similarities.argsort()[::-1][0:10]
    # for similar_values in similarities:
    #     print(similar_values)


    print(" Loading response from the model ..... ")
    new_df = data.loc[max_index]
    prompt = f'''
    I am Teaching Web Development Course using Sigma Web Development Course on Youtube 
    here are video chunks Containing Chunk id , start time , end time (seconds) and text along with it 
    {new_df[['chunk_id' , 'start_time' , 'end_time' , 'text']].to_json()}
    {question}
    User asked this question related to the video Chunks you have to answer in a human way (dont mention the above format it is just for reference) where and how much 
    content is taught where (in which video and at what timestamp) and guide the user to go to that 
    particular video
    if user asks unrelated questions tell them you can answer only related to the course 
    '''



    # Prompt Template 
    # prompt_template = ChatPromptTemplate.from_messages([
    #     ("system", "You are a helpful assistant for a video course"),
    #     ("user", prompt)
    # ])


    # chain = prompt_template | llm

    # # response = chain.invoke({})
    # response = chain.invoke({
    #     "user": prompt,
    #     "question": question
    # })


    response = llm.invoke(prompt)



    # new_dict = {}

    # print(new_df[['chunk_id','text']])
    for index , item in new_df.iterrows():
        print(index , item['chunk_id'] , item['text'])
    #     new_dict['index'] = index
    #     new_dict['text'] = item['text']
        
    # new_dict_df = pd.DataFrame.from_dict(new_dict)
    # print(new_dict)

    print(response.content)

    with open("response.txt" , 'w' , encoding='utf-8') as file:
        file.writelines(response.content)

print(" I hope You Loved Our Model ")