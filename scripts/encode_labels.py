import argparse
import os
import pickle

from datasets import load_dataset, load_from_disk
from tqdm.auto import tqdm
from transformers import AutoModel, AutoTokenizer

from audiodiffusion.audio_encoder import AudioEncoder
import torch

def get_vector_representation(input_string, model, tokenizer):
    input_tokens = tokenizer(input_string, return_tensors="pt")
    with torch.no_grad():
        encoder_outputs = model.encoder(input_tokens["input_ids"])
    vector_representation = encoder_outputs.last_hidden_state[:, 0, :]
    return vector_representation

def main(args):
    # audio_encoder = AudioEncoder.from_pretrained("teticio/audio-encoder")
    tokenizer = AutoTokenizer.from_pretrained("t5-3b")
    t5_model = AutoModel.from_pretrained("t5-3b")

    if args.dataset_name is not None:
        if os.path.exists(args.dataset_name):
            dataset = load_from_disk(args.dataset_name)["train"]
        else:
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                use_auth_token=True if args.use_auth_token else None,
                split="train",
            )

    encodings = {}
    df = dataset.to_pandas()
    for audio_file in tqdm(df["audio_file"]):
        # label = row["label"]
        label = audio_file.split("/")[-2]
        print("label is", label)
        
        encodings[audio_file] = get_vector_representation(label, model=t5_model, tokenizer=tokenizer)
    pickle.dump(encodings, open(args.output_file, "wb"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create pickled audio encodings for dataset of audio files.")
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--output_file", type=str, default="data/encodings.p")
    parser.add_argument("--use_auth_token", type=bool, default=False)
    args = parser.parse_args()
    main(args)
