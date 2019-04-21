#!/bin/bash

mkdir -p ./speaker_background_spectrograms

mkdir -p ./two_speakers_mix_spectrograms/
mkdir -p ./two_speakers_mix_spectrograms/part_1

mkdir -p ./speaker_video_embeddings/
mkdir -p ./speaker_video_embeddings/part_1

mkdir -p ./chatter_audios
mkdir -p ./chatter_audios/part_1
mkdir -p ./chatter_audios/part_2
mkdir -p ./chatter_audios/part_3

mkdir -p ./clean_audios
mkdir -p ./videos
mkdir -p ./frames
mkdir -p ./pretrained_model

mkdir -p temp/chatter_audio
mkdir -p temp/chatter_video

echo "Done setting up directories set"

python3 chatter_download.py
python3 chatter_slicer.py

rm -rf temp/*

echo "Done downloading chatter_files from audioset...."
echo "Now you can download avspeech dataset."
