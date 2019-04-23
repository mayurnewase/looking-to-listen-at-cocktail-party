from models import *
from model_utils import *
from audio_utils import *
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard
import argparse

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", type = int, default = 16)
	parser.add_argument("--epochs", type = int, default = 10)
	parser.add_argument("--lr", type = int, default = 0.001)
	parser.add_argument("--es_patience", type = int, default = 30)
	parser.add_argument("--model_type", type = str, default = "audio_model")  #audio_model/audio_video_model

	parser.add_argument("--filters", type = int, default = 32)
	parser.add_argument("--dropout", type = int, default = 0.2)
	parser.add_argument("--audio_shape", nargs='+', type = int) #only for audio_video_model, default = 298,257
	parser.add_argument("--video_shape", nargs="+", type = int) #only for audio_video_model, default = 75,512


	args = parser.parse_args()

	if args.model_type == "audio_model":
		model,db_model = AudioOnlyModel(filters = args.filters, dropout = args.dropout)
		model.compile(loss = loss, optimizer = Adam(lr = 0.0001), metrics = ["mse"])

	if args.model_type == "audio_video_model":
		vm = VideoModel(filters = args.filters, 
			audio_ip_shape = args.audio_shape, video_ip_shape = args.video_shape)

		#model = load_model("../input/video-model-2-speakers-big-audio/model.h5", custom_objects = {"tf" : tf, "loss" : loss})
		model.compile(loss = [loss, loss], optimizer = Adam(lr = 0.0001), metrics = ["mse"])

	batch_size = args.batch_size

	train_dic, val_dic = load_data(model_type)

	train_steps = int(np.floor(len(train_dic) / batch_size))
	val_steps = int(np.floor(len(val_dic) / batch_size))
	
	es = EarlyStopping(monitor='loss', min_delta=0, patience= args.es_patience, verbose=1, mode='min', baseline=None, restore_best_weights= True)
	rp = ReduceLROnPlateau(monitor='loss', factor=0.01, patience=5, verbose=1, mode='auto', min_delta=0.0001, cooldown=0, min_lr=0)
	tb = TensorBoard(log_dir='./tb_logs', histogram_freq=0, batch_size= batch_size, write_graph=True, write_grads=False, write_images=False,
	                 embeddings_freq=0, embeddings_layer_names=None,embeddings_metadata=None, embeddings_data=None, update_freq='epoch')

	tg = trainGen(batch_size, train_dic, train_steps)
	vg = valGen(batch_size, val_dic, val_steps)

	model.fit_generator(generator= tg, validation_data = vg, validation_steps = val_steps, epochs = 5, steps_per_epoch = train_steps, verbose = 1, shuffle = True,
	                   callbacks = [tb])

if __name__ == "__main__":
	main()



































