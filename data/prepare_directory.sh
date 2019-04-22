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

pip install -r requirements.txt

git clone https://github.com/davidsandberg/facenet.git data/facenet/
pip install face_recognition
sudo apt-get --assume-yes install ffmpeg
sudo apt-get --assume-yes --fix-missing install youtube-dl

rm -rf data/temp/*

wget https://storage.cloud.google.com/avspeech-files/avspeech_train.csv -P data/
wget https://storage.cloud.google.com/avspeech-files/avspeech_test.csv -P data/

echo "Done setting up directories and environment,now proceed to download dataset...."
