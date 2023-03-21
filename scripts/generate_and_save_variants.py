import re
import torch
import random
import librosa
import numpy as np
from datasets import load_dataset
from audiodiffusion import AudioDiffusion
import torchaudio
import os
import matplotlib.pyplot as plt
import scipy.io.wavfile as wavfile
import pandas as pd
from audiodiffusion.pipeline_audio_diffusion import AudioDiffusionPipeline

from transformers import AutoModel, AutoTokenizer
import torch

def get_vector_representation(input_string, model, tokenizer):
    input_tokens = tokenizer(input_string, return_tensors="pt")
    with torch.no_grad():
        encoder_outputs = model.encoder(input_tokens["input_ids"])
    vector_representation = encoder_outputs.last_hidden_state[:, 0, :]
    # vector_representation = torch.mean(encoder_outputs.last_hidden_state, dim=1) switch to me!
    return vector_representation

def get_train_audio_files(metadata_path: str, base_path: str = ""):
    df = pd.read_csv(metadata_path)

    paths = []
    for _, row in df.iterrows():
        paths.append(row['path'].split("data/")[-1])
    return paths


def generate_variants(model_id, start_step, input_dir, metadata_path = None, task_name: str = ""):
    augmentations_folder = "conditional_samples"
    if not os.path.exists("images"):
        os.makedirs("images")
    if not os.path.exists("audio"):
        os.makedirs("audio")
    
    if not os.path.exists(f"{augmentations_folder}/{task_name}"):
        os.makedirs(f"{augmentations_folder}/{task_name}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    generator = torch.Generator(device=device)

    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    t5_model = AutoModel.from_pretrained("t5-base").to()



    audio_diffusion = AudioDiffusion(model_id=model_id)
    # audio_diffusion.pipe.to("mps")
    mel = audio_diffusion.pipe.mel


    audio_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(input_dir)
        for file in files
        if re.search("\.(mp3|wav|m4a)$", file, re.IGNORECASE)
    ]
    print("starting audios", len(audio_files))
    if metadata_path:
        train_files = get_train_audio_files(metadata_path=metadata_path)
        audio_files = [file for file in audio_files if file.split("data/")[-1] in train_files]
        print("modified audios", len(audio_files))

    for i in range(1):
        for audio_file_path in audio_files:
            if "Common" in audio_file_path:
                continue
            # sr, audio_file = wavfile.read(audio_file_path)
            audio_file, sr = librosa.load(audio_file_path)
            print(f"sr {sr} and audio {audio_file}")
            label = audio_file_path.split("/")[-2]
            encoding = get_vector_representation(label, model=t5_model, tokenizer=tokenizer)
            print("label", label)

            seed = generator.seed()
            print(f'Seed = {seed}')
            generator.manual_seed(seed)
            image, (sample_rate,
                    audio) = audio_diffusion.generate_spectrogram_and_audio_from_audio(
                        raw_audio=audio_file, generator=generator, start_step = start_step, encoding=encoding)
            audio_name = audio_file_path.split('/')[-1]
            image_file = f"{augmentations_folder}/spectrogram_{audio_name}_{i}.png"
            image.save(image_file)
            audio_save_path = f"{augmentations_folder}/{task_name}/audio_{audio_name}_{i}.wav"
            if task_name == "watkins":
                species_name = audio_file_path.split('/')[-2]
                if not os.path.exists(f"{augmentations_folder}/{task_name}/{species_name}"):
                    os.makedirs(f"{augmentations_folder}/{task_name}/{species_name}")
                audio_save_path = f"{augmentations_folder}/{task_name}/{species_name}/audio_{audio_name}_{i}.wav"
            audio_save_path_og = f"{augmentations_folder}/{task_name}/{species_name}/audio_{audio_name}_{i}_og.wav"
            wavfile.write(audio_save_path, sample_rate, audio)
            wavfile.write(audio_save_path_og, sample_rate, audio_file)
    
if __name__ == "__main__":
    generate_variants(model_id = "models/watkins_conditional_80", start_step=0, input_dir="../beans/data/watkins",
                        metadata_path="../beans/data/watkins/annotations.train.csv", task_name = "watkins")
