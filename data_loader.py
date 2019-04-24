from random import shuffle


class load_data():

	def loadAudioData(self, audio_spec_path):
		files = os.listdir(audio_spec_path)
		
		train_clean = []
		train_mix = []

		for file in file:
			subfiles = os.listdir(audio_spec_path + file):
			for subfile in subfiles:
				if subfile != file + ".pkl":
					train_clean.append(audio_spec_path + file + "./pkl")
					train_mix.append(audio_spec_path + file + "/" + subfile)


		combo_train = list(zip(train_clean, train_mix))
		shuffle(combo_train)
		train_clean, train_mix = zip(*combo_train)

		combo_val = list(zip(val_clean, val_mix))
		shuffle(combo_val)
		val_clean, val_mix =  zip(*combo_val)

		return train_clean, train_mix


	def loadAudioVideoData(self):
		pass		

class generators():

	def __init__(self, batch_size):
		self.batch_size = batch_size

	def trainAudioGen():
	    #for i in range(3):  # for epochs
	    train_steps = int(np.floor(len(train_clean[:train_size]) / self.batch_size))

	    while 1:
	        for step in range (num_steps):
	            x = []
	            y = []

	            clean_batch = train_clean[step * self.batch_size : (step+ 1) * self.batch_size]
	            mix_batch = train_mix[step * self.batch_size : (step+ 1)* self.batch_size]
	            for clean_file, mix_file in zip(clean_batch, mix_batch):
	                x.append(pickle.load(open(mix_file, "rb")))
	                y.append(pickle.load(open(clean_file, "rb")))

	            x_arr = np.array(x)
	            y_arr = np.array(y)

	            yield x_arr, y_arr

	def valAudioGen():
		#for i in range(3):  # for epochs
		val_steps = int(np.floor(len(val_clean) / self.batch_size))
	    while 1:
	        for step in range (num_steps):
	            x = []
	            y = []

	            clean_batch = val_clean[step * self.batch_size : (step+ 1) * self.batch_size]
	            mix_batch = val_mix[step * self.batch_size : (step+ 1)* self.batch_size]
	            for clean_file, mix_file in zip(clean_batch, mix_batch):
	                x.append(pickle.load(open(mix_file, "rb")))
	                y.append(pickle.load(open(clean_file, "rb")))

	            x_arr = np.array(x)
	            y_arr = np.array(y)

	            yield x_arr, y_arr		















































