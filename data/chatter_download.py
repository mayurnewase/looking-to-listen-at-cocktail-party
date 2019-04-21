"""
read file
get id
download clip
save file with same id as name
"""

import pandas as pd
import os
import subprocess

video_dataset_path = "data/temp/chatter_video/"
audio_dataset_path = "data/temp/chatter_audio/"

chatter_tag = "/m/07rkbfh"

data = pd.read_csv("unbalanced_train_segments.csv", header = None, names = ["id", "start", "end", "tag"])

for i in range(data.shape[0]):

	if ((chatter_tag in data.loc[i, "tag"]) and (not os.path.isfile(audio_dataset_path + data.loc[i, "id"] + ".wav"))):
		print("downloading", data.loc[i, "id"], data.loc[i, "start"], data.loc[i, "end"])

		url = "youtube-dl -f best --get-url https://www.youtube.com/watch?v=" + str(data.loc[i, "id"])
		res1 = subprocess.run(url, stdout = subprocess.PIPE, shell=True).stdout.decode("utf-8").rstrip()
		
		download = "ffmpeg" + " -ss " + str(data.loc[i,"start"]) + " -i \"" + res1 + "\"" + " -t " + str(data.loc[i, "end"] - data.loc[i, "start"]) + " -c:v copy -c:a copy " + video_dataset_path + str(data.loc[i,"id"]) +".mp4"
		print("command \n", download)

		res2 = subprocess.Popen(download, stdout = subprocess.PIPE, shell=True).communicate()
		
		os.popen("ffmpeg -i " + video_dataset_path + data.loc[i, "id"] + ".mp4" + " -vn " + audio_dataset_path + data.loc[i, "id"] + ".wav").read()

		if(os.path.isfile(video_dataset_path + data.loc[i, "id"] + ".mp4")):
			os.remove(video_dataset_path + data.loc[i, "id"] + ".mp4")

	else:
		print("skppiing", i)














































