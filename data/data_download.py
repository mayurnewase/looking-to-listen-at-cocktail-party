import pandas as pd
import os
import subprocess
import argparse

from speaker_background import *
from mix_speakers import *
from video_generator import *

def main():

	videos_path = "data/videos/"
	audio_dataset_path = "data/clean_audios/"

	data = pd.read_csv("data/avspeech_train.csv", header = None, names = ["id", "start", "end", "x", "y"])
	print(data.head())

	parser = argparse.ArgumentParser()
	parser.add_argument("--from_id", type = int, default = 0)
	parser.add_argument("--to_id", type = int, default = 20000)

	parser.add_argument("--type_of_dataset", type = str, default = "audio_dataset")  #audio_dataset / audio_video_dataset
	parser.add_argument("--low_memory", type = str, default = "no")

	parser.add_argument("--chatter_part", type = int, default = 1)
	parser.add_argument("--video_part", type = int, default = 1)

	parser.add_argument("--sample_rate", type = int, default = 16000)
	parser.add_argument("--duration", type = int, default = 3)
	parser.add_argument("--mono", type = str, default = True)
	parser.add_argument("--window", type = int, default = 400)
	parser.add_argument("--stride", type = int, default = 160)
	parser.add_argument("--fft_length", type = int, default = 512)
	parser.add_argument("--amp_norm", type = int, default = 0.3)
	parser.add_argument("--chatter_norm", type = int, default = 0.3)

	parser.add_argument("--face_extraction_model", type = str, default = "cnn")
	args = parser.parse_args()

	if args.type_of_dataset == "audio_dataset":
		sb = SpeakerBackground(chatter_part = args.chatter_part, sample_rate = args.sample_rate, 
			duration = args.duration, mono = args.mono, window = args.window, stride = args.window, fft_length = args.fft_length, amp_norm = args.amp_norm, 
			chatter_norm = args.chatter_norm)
	
	if args.type_of_dataset == "audio_video_dataset":
		ms = MixSpeakers(sample_rate = args.sample_rate, duration = args.duration, mono = args.mono, window = args.window,
			stride = args.stride, fft_length = args.fft_length, amp_norm = args.amp_norm)

		vs = VideoExtract(args.from_id, args.to_id, args.fps, args.duration, args.video_part, face_extraction_model)

	for i in range(args.from_id, args.to_id):

		if (not os.path.isfile(videos_path + data.loc[i, "id"] + ".mp4")):           #CHECK FOR AUDIO DIRECTORY ALSO -> VIDEO IS DELETED COZ IT WAS CONVERTED........
			
			print("downloading", data.loc[i, "id"], data.loc[i, "start"], data.loc[i, "end"])
			url = "youtube-dl -f best --get-url https://www.youtube.com/watch?v=" + str(data.loc[i, "id"])		#download youtube video
			res1 = subprocess.run(url, stdout = subprocess.PIPE, shell=True).stdout.decode("utf-8").rstrip()
			if(res1 == ""):
				print("----------------------Video not available---------------------")
				continue

			#trim video
			download = "ffmpeg" + " -ss " + str(data.loc[i,"start"]) + " -i \"" + res1 + "\"" + " -t " + str(float(data.loc[i, "end"]) - float(data.loc[i, "start"])) + " -c:v copy -c:a copy " + videos_path + str(data.loc[i,"id"]) + ".mp4"
			res2 = subprocess.Popen(download, stdout = subprocess.PIPE, shell=True).communicate()
		
		else:
			print("skppiing", i)

		#video downloaded in orig_dataset/
		error = sb.extract_wav(data.loc[i, "id"])   #speaker + background in all_spectrograms
		
		if error == 1:
			print("========extract wav failed skipping mixing speakers or video extraction=======")
			continue

		sb.mix_speakers(data.loc[i, "id"])  #speaker + speaker in mixed_speakers
		result = vs.extract_video(data.loc[i, "id"], data.loc[i, "x"], data.loc[i, "y"])  #extract video embeddings in speaker_clean_video

		if result == 1 :
			print("video embedding extraction failed see logs for reason")

	print("Done creating dataset")
	if (args.low_memory == "yes"):
		delete_audios = "rm {0}*".format(self.audio_dataset_path)
		delete_videos = "rm {0}*".format(self.videos_path)
		rs = subprocess.Popen(delete_audios, stdout = subprocess.PIPE, shell = True).communicate()	
		rs = subprocess.Popen(delete_videos, stdout = subprocess.PIPE, shell = True).communicate()	

if __name__ == "__main__":
	main()









































