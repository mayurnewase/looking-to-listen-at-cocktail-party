#ffmpeg -i orig_dataset/_8K1hWkirLo.mp4 -vn clean_audio/_8K1hWkirLo.wav
#Bug maybe  -> noise file is resampled by librosa -> whole mix is resampled by tensorflow for spec but not noise file individually

import os
import librosa
import functools
import pickle
import shutil

import tensorflow as tf
from tensorflow.python.ops import io_ops

class SpeakerBackground():

	def __init__(self, chatter_part = 1, sample_rate = 16000, duration = 3, mono = True, window = 400, stride = 160, fft_length = 512, amp_norm = 0.3, 
		chatter_norm = 0.3):

		self.orig_dataset = "data/videos/"
		self.clean_audio = "data/clean_audios/"
		self.spect_path = "data/speaker_background_spectrograms/"
		self.chatter_path = "data/chatter_audios/"

		self.chatter_files = [f for f in os.listdir(self.chatter_path) if os.path.isfile(os.path.join(self.chatter_path, f))]
		self.chatter_part = chatter_part

		self.sample_rate = sample_rate
		self.duration = duration
		self.mono = mono

		self.window = window  #in samples directly
		self.stride = stride
		self.fft_length = fft_length
		self.amp_norm = amp_norm
		self.chatter_norm = chatter_norm

	def find_spec(self, filename, mixing, mix_name = None):
		print("-------------finding spectrogram for {filename}----------------")
		with tf.Session(graph=tf.Graph()) as sess:

			holder = tf.placeholder(tf.string, [])
			file = tf.read_file(holder)
			decoder = tf.contrib.ffmpeg.decode_audio(file, file_format = "wav", samples_per_second = self.sample_rate, channel_count = 1)

			stft = tf.signal.stft(tf.transpose(decoder), frame_length = self.window, frame_step = self.stride, fft_length = self.fft_length, window_fn = tf.signal.hann_window)

			amp = tf.squeeze(tf.abs(stft)) ** self.amp_norm
			phase = tf.squeeze(tf.angle(stft))

			stacked = tf.stack([amp, phase], 2)

			if mixing == 0:
				stft = sess.run(stacked, feed_dict = {holder : self.spect_path + filename + "/" + filename + ".wav"})
				pickle.dump(stft, open(self.spect_path + filename + "/" + filename  + ".pkl", "wb"))
			else:
				stft = sess.run(stacked, feed_dict = {holder : self.spect_path + filename + "/" + mix_name + ".wav"})
				pickle.dump(stft, open(self.spect_path + filename + "/" + mix_name + ".pkl", "wb"))
			print("============STFT SHAPE IS {0}=============".format(stft.shape))

	def mix_audio(self, filename):
		print("-----------mixing audio---------------")

		for index, chatter in enumerate(self.chatter_files):
			print("mixing {0} file".format(index))

			f1, _ = librosa.load(self.spect_path + filename + "/" + filename + ".wav", sr = self.sample_rate, mono = self.mono, duration = self.duration)  #load clean file
			f2, _ = librosa.load(self.chatter_path + "chatter_" + str(self.chatter_slot) + "/" + chatter, sr = self.sample_rate, mono = self.mono, duration = self.duration)		#load chatter file

			combo = f1 + (chatter_norm * f2)
			
			mix_name = filename + "_mix_" + str(index)
			librosa.output.write_wav(self.spect_path + filename + "/" + mix_name + ".wav", combo , sr = self.sample_rate , norm = False)     #DELETE MIX FILE -> stored mix in seperate directory

			find_spec(filename, 1, mix_name)
			os.remove(self.spect_path + filename + "/" + mix_name + ".wav")   #delete mixed wav after getting spec


	def extract_wav(self, filename):
		print("-----------extracting audio-------------")
		wavfile = filename + ".wav"

		if (not os.path.isdir(self.spect_path + filename)):

			os.mkdir(self.spect_path + filename)
			os.popen("ffmpeg -i " + self.orig_dataset + filename + ".mp4" + " -vn " + self.spect_path + filename + "/" + wavfile).read()		#extract audio
			
			if(not os.path.isfile(self.spect_path + filename + "/" + wavfile)):
				print("----------------ffmpeg can't extract audio so deleting directory--------------")
				shutil.rmtree(self.spect_path + filename)
				return 1

			data, _ = librosa.load(self.spect_path + filename + "/" + wavfile, sr = self.sample_rate, mono = self.mono, duration = self.duration)
			librosa.output.write_wav(self.spect_path + filename + "/" + wavfile, data, self.sample_rate, norm = False)    #Delete this file in end
			
			#Used for mixing speakers later if required so don't delete this.
			librosa.output.write_wav(self.clean_audio +  wavfile, data, self.sample_rate, norm = False)    

			find_spec(filename , 0)
			mix_audio(filename)
			os.remove(self.spect_path + filename + "/" + wavfile)		#remove clean file from spect dir after mixing with all chatters

		else:
			print("skipping audio extraction for {0}".format(filename))







































































































