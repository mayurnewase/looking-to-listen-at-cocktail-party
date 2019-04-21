import os
import librosa
import functools
import pickle
import shutil
import numpy as np

op_chatter_path = "chatter_audios/"    #chatter_3_seconds
ip_chatter_path = "temp/chatter_audio/"

sample_rate = 16000
duration = 3
mono = True

def divide_chatter(audio_path, audio_file, limit_slots = 5):
	full_audio,_ = librosa.load(audio_path + audio_file, sr = sample_rate, mono = mono)
	file_duration = np.floor(full_audio.shape[0] / sample_rate)
	slots = int(np.floor(file_duration / 3))
	if slots > limit_slots:
		print("slots more than {0} available in {1} limiting to {0}".format(limit_slots, audio_file))
		slots = limit_slots


	for slot in range(0, slots):
		file_slotted = full_audio[slot * sample_rate * duration : (slot+1) * sample_rate * duration]
		print(file_slotted.shape, "{}chatter_{}".format(op_chatter_path, slot+1))
		librosa.output.write_wav("{0}part_{1}/{2}".format(op_chatter_path, slot+1 , audio_file), file_slotted, sr = sample_rate, norm = False)


chatter_files = [f for f in os.listdir(ip_chatter_path) if os.path.isfile(os.path.join(ip_chatter_path, f))]

for file in chatter_files:
	print("slotting {}".format(file))
	divide_chatter(ip_chatter_path, file)














