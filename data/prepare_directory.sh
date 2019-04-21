#!/bin/bash

mkdir -p data/speaker_background_spectrograms

mkdir -p data/two_speakers_mix_spectrograms/
mkdir -p data/two_speakers_mix_spectrograms/part_1

mkdir -p data/speaker_video_embeddings/
mkdir -p data/speaker_video_embeddings/part_1

mkdir -p data/chatter_audios
mkdir -p data/chatter_audios/part_1
mkdir -p data/chatter_audios/part_2
mkdir -p data/chatter_audios/part_3

mkdir -p data/clean_audios
mkdir -p data/videos
mkdir -p data/frames
mkdir -p data/pretrained_model

mkdir -p data/temp/chatter_audio
mkdir -p data/temp/chatter_video

echo "Done setting up directories set"

python3 data/chatter_download.py
python3 data/chatter_slicer.py

rm -rf data/temp/*

echo "Done downloading chatter_files from audioset...."
echo "Now you can download avspeech dataset."
