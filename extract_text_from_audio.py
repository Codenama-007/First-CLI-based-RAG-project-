from faster_whisper import WhisperModel
import json
import os

print("Loading model...")

model = WhisperModel("small", compute_type="int8")

print("Model loaded successfully.")

folder_path = "audios"
output_folder = "transcriptions"

# Create output folder if not exists
os.makedirs(output_folder, exist_ok=True)
print(" transciptions folder created successfully.")

# Supported audio formats
audio_extensions = (".mp3", ".wav", ".m4a")

for file in os.listdir(folder_path):

    # Skip non-audio files
    if not file.endswith(audio_extensions):
        continue

    print(f"Transcribing file: {file}")

    file_path = os.path.join(folder_path, file)

    segments, info = model.transcribe(
        file_path,
        beam_size=5 ,
        task = "translate"
    )

    results = []

    for i, segment in enumerate(segments):

        results.append({
            "chunk_id": f"{file}_{i+1:03}",
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip() ,
            # 'entire_text' : segment.text
        })

    # Remove extension from filename
    filename_without_ext = os.path.splitext(file)[0]

    output_path = os.path.join(
        output_folder,
        f"{filename_without_ext}.json"
    )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    print(f"{file} successfully transcribed.")
    print(f"Saved to: {output_path}")

print("All files processed.")