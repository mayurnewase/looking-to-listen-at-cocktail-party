"""
data generation
	train.csv
	val.csv

	db structure:
		speaker_background_spectrograms/
			per speaker part 1/
				speaker_clean.pkl
				speaker_chatter_i.pkl

			per speaker part 2/
				speaker_clean.pkl
				apeaker_chatter_i.pkl

		two_speakers_mix_spectrograms/
			per speaker/
				clean.pkl
				mix_with_other_i.pkl

		speaker_video_spectrograms
			per_speaker part 1//
				clean.pkl

			per_speaker part 2/
				clean.pkl

		chatter audios/
			part1/
			part2/
			part3/

		clean audios/
		videos/
		frames/
		pretrained_model/
			facenet_model.h5
	
	if save memory:
		clean those after processing

model
train
validation results
-----------------------------
add chatter slicer

------------------------------
STEPS:

	install requirements
	./prepare_directory

	!git clone https://github.com/davidsandberg/facenet.git
	!pip install face_recognition
	!sudo apt-get --assume-yes install ffmpeg
	sudo apt-get install youtube-dl




"""




































