![alt title](https://github.com/mayurnewase/looking-to-listen-at-cocktail-party/blob/master/assets/logo.png)

This is Keras+Tensorflow implementation of paper ["Looking to listen at the cocktail party: A speaker-independent audio-visual model for speech separation"](https://arxiv.org/abs/1804.03619)
from Ephrat et el. from Google Research.
The project also uses ideas from the paper ["Seeing Through Noise:Visually Driven Speaker Seperation and Enhancement"](https://arxiv.org/pdf/1708.06767.pdf)

## Compatibility
The code is tested using Tensorflow 1.13.1 under Ubuntu 18.00 with python 3.6.

## News
| Date     | Update |
|----------|--------|
| 08-06-2019 | Notebook added for full pipeline with pretrained model. |
| 25-05-2019 | Datasets added for mixed user videos. |
| 23-04-2019 | Added automated scripts for creating database structure. |

## Usage
### Database structure
Given a way to store audio and video datasets efficiently without much duplication.
```
|--speaker_background_spectrograms/
|  |--per speaker part 1/
|  |    |--speaker_clean.pkl
|  |    |--speaker_chatter_i.pkl
|  |--per speaker part 2/
|  |--  |--speaker_clean.pkl
|       |--speaker_chatter_i.pkl
|--two_speakers_mix_spectrograms/
|	 |--per speaker/
|	 |	|--clean.pkl
|	 |	|--mix_with_other_i.pkl
|--speaker_video_spectrograms
|	 |--per_speaker part 1/
|	 |	|--clean.pkl
|	 |--per_speaker part 2/
|	 |	|--clean.pkl
|--chatter audios/
|  |--part1/
|  |--part2/
|  |--part3/
|--clean audios/
|	 |--videos/
|	 |--frames/
|	 |--pretrained_model/
|	 |  |--facenet_model.h5
```

### Getting started
1.Install all dependencies
```
pip install -r requirements.txt
```
2.Run prepare_directory script
```
./data/prepare_directory.sh
```
3.download avspeech train and test [csv files](https://looking-to-listen.github.io/avspeech/download.html) and put in data/

4.Run background chatter files downloader and slicer to download and slice chatter files.This will download chatter files with tag "/m/07rkbfh" from [Audioset](https://research.google.com/audioset/index.html)
```
python data/chatter_download.py
python data/chatter_slicer.py
```
5.Start Downloading data for avspeech dataset and process with your choice with arguments.
```
python data/data_data_download.py --from_id=0 --to_id=1000 --type_of_dataset=audio_dataset
```
Arguments available
```
from_id -> start downloading youtube clips download from train.csv from this id

to_id -> start downloading youtube clips download from train.csv to this id

type_of_dataset -> type of dataset to prepare.
  audio_dataset -> create audio spectrogram mixed with background chatter
  audio_video_dataset -> create audio spectrogram and video embeddings and spectrograms of speaker mixed other speakers audio.
  
low_memory -> clear unnecessary stuff

chatter_part -> user different slots of chatter files to be mixed with clean speakers audio

sample_rate,duration,fps,mono,window,stride,fft_length,amp_norm,chatter_norm -> arguments for STFT and audio processing

face_extraction_model -> select which model to use for facial embedding extraction
  hog -> faster on cpu but less accurate
  cnn -> slower on cpu,faster on nvidia gpu,more accurate
```

### Datasets
1. Video mixed dataset is availble on my kaggle page in 10 parts.(created using default parameters above)
```
Go to my kaggle profile(https://www.kaggle.com/mayurnewase)
Click on datasets
Sort by new
Datasets are named by mix_speakers_ultimate_*
Total 10 parts are available.
```

## To do
Check [here](https://github.com/users/mayurnewase/projects/2)

