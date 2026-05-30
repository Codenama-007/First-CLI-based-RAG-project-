Files Used :- 
extract_text_from_audio.py -> It is a program to convert all the Audio Files into Json Files .
we get a transcriptions in the json form 

chunking_to_embedding.py -> This program will basically make embeddings related to Chunks that i just Provided in the form of json 

process.py -> This file contains LLM model and it gives reponse with respect to Chunks that i just made 


<!-- Steps to Wite the Program  -->

<!-- Step 1 --> 
load all the Audio files in a specific Audio Folder and perform Transcriptions . You can also make openai API keys or any API key that you know . However i loaded the Model Faster Whisper model on my system 

<!-- Step 2 -->
Second Step is that After All your Transcripions is done it should give a Transcriptions Folder with all the Transcriptions in it 

<!-- Step 3 -->
Once all the Transcriptions are done you would want to Make Embeddings out of those Chunks . These Embeddings are then Stored along with Video title , start time , end time and text

<!-- Step 4 -->
After The embeddings part you can use sklearn to dervive Similarity, ollama models through Langchain to make a Proper Response out of it 


<!-- Pro Tip -->
You Can Upload Video follow the methodology given below 

Video -> Audio -> Chunking -> Embedding 

## i given a demo audio and its transcription file 
## you need a separate Audio folder and its Transcription folder for storing all the Transcriptions