from models import *
from utils import *
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard

def main():

	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", type = int, default = 16)
	parser.add_argument("--epochs", type = int, default = 10)
	parser.add_argument("--lr", type = int, default = 0.001)
	parser.add_argument("--es_patience", type = int, default = 30)

	args = parser.parse_args()

	model = FullModel()
	#model = load_model("../input/video-model-2-speakers-big-audio/model.h5", custom_objects = {"tf" : tf, "loss" : loss})
	model.compile(loss = [loss, loss], optimizer = Adam(lr = 0.0001), metrics = ["mse"])

	batch_size = args.batch_size

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



































