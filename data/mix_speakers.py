"""
for each audio:
	create new dir
	save clean spectro
	read all dirs present except itself
	for each dir read:
		read that audio
		mix with current
		save spec in that dir

a a ab ac ad 
b b bc bd
c c cd
d d
"""
import os
import librosa
import functools
import pickle
import shutil

import tensorflow as tf
from tensorflow.python.ops import io_ops

class MixSpeakers():

	def __init__(self, sample_rate, duration, mono, window, stride, fft_length, amp_norm):
		self.sample_rate = sample_rate
		self.duration = duration
		self.mono = mono

		self.window = window
		self.stride = stride
		self.fft_length = fft_length
		self.amp_norm = amp_norm

		self.clean_audio = "clean_audio/"
		self.mix_speakers_dir = "two_speakers_mix_spectrograms/"

		
		if not os.path.isdir(self.clean_audio):
			os.mkdir(self.clean_audio)

		if not os.path.isdir(self.mix_speakers_dir):
			os.mkdir(self.mix_speakers_dir)


	def find_spec(self, speaker_id, src_file , destination_file):
		print("-------------finding spectro----------------")
		with tf.Session(graph=tf.Graph()) as sess:
			holder = tf.placeholder(tf.string, [])
			file = tf.read_file(holder)
			decoder = tf.contrib.ffmpeg.decode_audio(file, file_format = "wav", samples_per_second = self.sample_rate, channel_count = 1)

			stft = tf.signal.stft(tf.transpose(decoder), frame_length = self.window, frame_step = self.stride, fft_length = self.fft_length, window_fn = tf.signal.hann_window)

			amp = tf.squeeze(tf.abs(stft)) ** self.amp_norm
			phase = tf.squeeze(tf.angle(stft))

			stacked = tf.stack([amp, phase], 2)

			stft = sess.run(stacked, feed_dict = {holder : src_file})
			pickle.dump(stft, open(destination_file, "wb"))
			print("============STFT SHAPE IS {0}=============".format(stft.shape))

	def mix_speakers(self, speaker_id):    #take current speaker id to mix with other speakers

		print("-------Mixing {} speakers------".format(speaker_id))

		if not os.path.isdir(self.mix_speakers_dir + speaker_id):

			all_speakers = os.listdir(self.mix_speakers_dir)

			os.mkdir(self.mix_speakers_dir + speaker_id)
			self.find_spec(speaker_id, src_file = self.clean_audio + speaker_id + ".wav" , destination_file = self.mix_speakers_dir + speaker_id + "/" + speaker_id + ".pkl") 

			curr_audio, _ = librosa.load(self.clean_audio + speaker_id + ".wav", sr = self.sample_rate, mono = self.mono, duration = self.duration)

			for old_speaker in all_speakers:
				print("mixing {} and  {}".format(speaker_id, old_speaker))

				old_audio, _ = librosa.load(self.clean_audio + old_speaker + ".wav", sr = self.sample_rate, mono = self.mono, duration = self.duration)

				mixed_audio = curr_audio + old_audio
				librosa.output.write_wav(self.mix_speakers_dir+ old_speaker+"/" +old_speaker+ "_" +speaker_id+ ".wav", mixed_audio, self.sample_rate, norm = False)  #delete this file

				self.find_spec(old_speaker, src_file = self.mix_speakers_dir+ old_speaker+ "/"+ old_speaker+ "_" +speaker_id+ ".wav", destination_file= self.mix_speakers_dir+ old_speaker+"/" + old_speaker+ "_" + speaker_id + ".pkl")

				os.remove(self.mix_speakers_dir+ old_speaker+ "/"+ old_speaker+ "_" +speaker_id+ ".wav")










































