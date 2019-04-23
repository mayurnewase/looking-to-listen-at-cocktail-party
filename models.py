from keras.layers import *
from keras import Model
import keras.backend as K
from keras.optimizers import Adam
from keras.models import load_model
from keras.callbacks import TensorBoard

import tensorflow as tf


def AudioOnlyModel(filters = 32, dropout = 0.2, audio_shape = [298, 257]):
    #298 is temporal dimension
    #don't use for loss calculation
    ip = Input(shape = (audio_shape[0], audio_shape[1], 2)) ; print("input", ip.shape)
    ip_magnitude = Lambda(lambda x : x[:,:,:,0],name="ip_mag")(ip); print("ip_mag ", ip_magnitude.shape)  #takes magnitude from stack[magnitude,phase]
    ip_phase = Lambda(lambda x : tf.expand_dims(x[:,:,:,1], axis = -1),name="ip_phase")(ip); print("ip_phase ", ip_phase.shape)  #takes phase from stack[magnitude,phase]
       
    conv = Conv2D(filters = filters, kernel_size = (3,3), strides = (1,1), padding = "same", dilation_rate = (1,1),
                  activation = "relu")(ip) ; print("conv ", conv.shape)
    conv = BatchNormalization(axis=-1)(conv)
    #conv = SpatialDropout2D(rate = dropout)(conv)
    conv = Dropout(rate = dropout)(conv)
    
    conv = Conv2D(filters = filters, kernel_size = (3,3), strides = (1,1), padding = "same", dilation_rate = (1,1),
                  activation = "relu")(conv) ; print("conv ", conv.shape)
    conv = BatchNormalization(axis=-1)(conv)
    #conv = SpatialDropout2D(rate = dropout)(conv)
    conv = Dropout(rate = dropout)(conv)
    
    conv = Conv2D(filters = filters, kernel_size = (3,3), strides = (1,1), padding = "same", dilation_rate = (1,1),
                  activation = "relu")(conv) ; print("conv ", conv.shape)
    conv = BatchNormalization(axis=-1)(conv)
    #conv = SpatialDropout2D(rate = dropout)(conv)
    conv = Dropout(rate = dropout)(conv)
    
    conv = Conv2D(filters = filters* 2, kernel_size = (3,3), strides = (1,1), padding = "same", dilation_rate = (1,1),
                  activation = "relu")(conv) ; print("conv ", conv.shape)
    conv = BatchNormalization(axis=-1)(conv)
    #conv = SpatialDropout2D(rate = dropout)(conv)
    conv = Dropout(rate = dropout)(conv)
    
    conv = Conv2D(filters = filters* 2, kernel_size = (3,3), strides = (1,1), padding = "same", dilation_rate = (1,1),
                  activation = "relu")(conv) ; print("conv ", conv.shape)
    conv = BatchNormalization(axis=-1)(conv)
    #conv = SpatialDropout2D(rate = dropout)(conv)
    conv = Dropout(rate = dropout)(conv)
    
    conv = Conv2D(filters = filters* 3, kernel_size = (3,3), strides = (1,1), padding = "same", dilation_rate = (1,1),
                  activation = "relu")(conv) ; print("conv ", conv.shape)
    conv = BatchNormalization(axis=-1)(conv)
    #conv = SpatialDropout2D(rate = dropout)(conv)
    conv = Dropout(rate = dropout)(conv)
    
    conv = Conv2D(filters = filters* 3, kernel_size = (5,5), strides = (1,1), padding = "same", dilation_rate = (1,1),
                  activation = "relu")(conv) ; print("conv ", conv.shape)
    conv = BatchNormalization(axis=-1)(conv)
    #conv = SpatialDropout2D(rate = dropout)(conv)
    conv = Dropout(rate = dropout)(conv)
    
    conv = Conv2D(filters = filters* 3, kernel_size = (5,5), strides = (1,1), padding = "same", dilation_rate = (1,1),
                  activation = "relu")(conv) ; print("conv ", conv.shape)
    conv = BatchNormalization(axis=-1)(conv)
    #conv = SpatialDropout2D(rate = dropout)(conv)
    conv = Dropout(rate = dropout)(conv)
    
    conv = Conv2D(filters = filters* 3, kernel_size = (5,5), strides = (1,1), padding = "same", dilation_rate = (1,1),
                  activation = "relu")(conv) ; print("conv ", conv.shape)
    conv = BatchNormalization(axis=-1)(conv)
    #conv = SpatialDropout2D(rate = dropout)(conv)
    conv = Dropout(rate = dropout)(conv)

    #shape = tf.shape(conv)# fault here...add like timedistributed flatten dense
    #reshape1 = Lambda(lambda x : tf.reshape(x, [shape[0], data_shape[0], data_shape[1] * 8]), name = "reshape1")(conv) ; print("reshape1", reshape1.shape)
    reshape1 = TimeDistributed(Flatten())(conv)

    lstm = Bidirectional(LSTM(units = 32, return_sequences = True, activation = "tanh", dropout= dropout))(reshape1)   ;print("lstm", lstm.shape)
    
    #dense = TimeDistributed(Dense(data_shape[1], activation = "sigmoid"))(lstm)
    #output_mag = Lambda(lambda x : tf.multiply(x[0], x[1]), name = "mask_multiply")([ip_magnitude, dense]) ; print("output_mag", output_mag.shape)
    
    flatten = Flatten()(lstm) ;print("flatten ", flatten.shape)
    
    #dense = Dense(data_shape[0] * data_shape[1], activation = "sigmoid")(flatten) ; print("dense1", dense.shape)  #main
    
    dense = Dense(200, activation = "relu")(flatten)
    dense = Dropout(rate = dropout)(dense)
    #dense = Dropout(dropout)(dense)
    #dense = Dense(200, activation = "relu")(dense)
    #dense = Dropout(dropout)(dense)
    #dense = Dense(600, activation =  "relu")(dense)
    #dense = Dense(256, activation =  "tanh")(dense)
    dense = Dense(audio_shape[0] * audio_shape[1], activation = "sigmoid")(dense)
    
    reshape = Reshape([audio_shape[0], audio_shape[1]])(dense) ; print("reshape ", reshape.shape)  #mask
    output_mag = Lambda(lambda x : tf.multiply(x[0], x[1]), name = "mask_multiply")([ip_magnitude, reshape]) ; print("output_mag", output_mag.shape)
    output_mag = Lambda(lambda x : tf.expand_dims(x, axis= -1), name= "expand_dim")(output_mag) ; print("output_mag_expand", output_mag.shape)
    
    output_total = Lambda(lambda x : tf.concat(values=[x[0], x[1]], axis = -1),name="concat_mag_phase")([output_mag, ip_phase]) ; print("output_total ", output_total.shape)
    model = Model(ip, [output_total])
    debug_model = Model(ip, [reshape, output_mag])
    return model, debug_model


class VideoModel():

	def __init__(self, filters, audio_ip_shape, video_ip_shape):

		self.filters = filters
		self.audio_ip_shape = audio_ip_shape
		self.video_ip_shape = video_ip_shape

		self.conv1 = Conv1D(filters = filters, kernel_size = (7), padding = "same", dilation_rate = (1),
		              activation = "relu")
		self.bn1 = BatchNormalization(axis=-1)

		self.conv2 = Conv1D(filters = filters, kernel_size = (5), padding = "same", dilation_rate = (1),
		              activation = "relu")
		self.bn2 = BatchNormalization(axis=-1)

		self.conv3 = Conv1D(filters = filters, kernel_size = (5), padding = "same", dilation_rate = (2),
		              activation = "relu")
		self.bn3 = BatchNormalization(axis=-1)

		self.conv4 = Conv1D(filters = filters, kernel_size = (5), padding = "same", dilation_rate = (4),
		              activation = "relu")
		self.bn4 = BatchNormalization(axis=-1)

		self.conv5 = Conv1D(filters = filters, kernel_size = (5), padding = "same", dilation_rate = (8),
		              activation = "relu")
		self.bn5 = BatchNormalization(axis=-1)

		self.conv6 = Conv1D(filters = filters, kernel_size = (5), padding = "same", dilation_rate = (16),
		              activation = "relu")
		self.bn6 = BatchNormalization(axis=-1)

		self.conv7 = Lambda(lambda x : tf.expand_dims(x, axis = -1))

		self.conv8 = Lambda(lambda x: tf.image.resize_nearest_neighbor(x, size = (298, x.shape[-2])))

	def AudioModel(ip, filters = 32, dropout = 0.2):
	    
	    conv = Conv2D(filters = filters, kernel_size = (3,3), strides = (1,1), padding = "same", dilation_rate = (1,1),
	                  activation = "relu")(ip) ; print("conv ", conv.shape)
	    conv = BatchNormalization(axis=-1)(conv)
	    #conv = SpatialDropout2D(rate = dropout)(conv)
	    
	    conv = Conv2D(filters = filters, kernel_size = (3,3), strides = (1,1), padding = "same", dilation_rate = (1,1),
	                  activation = "relu")(conv) ; print("conv ", conv.shape)
	    conv = BatchNormalization(axis=-1)(conv)
	    #conv = SpatialDropout2D(rate = dropout)(conv)
	    
	    conv = Conv2D(filters = filters, kernel_size = (3,3), strides = (1,1), padding = "same", dilation_rate = (1,1),
	                  activation = "relu")(conv) ; print("conv ", conv.shape)
	    conv = BatchNormalization(axis=-1)(conv)
	    #conv = SpatialDropout2D(rate = dropout)(conv)
	    
	    conv = Conv2D(filters = filters* 2, kernel_size = (3,3), strides = (1,1), padding = "same", dilation_rate = (1,1),
	                  activation = "relu")(conv) ; print("conv ", conv.shape)
	    conv = BatchNormalization(axis=-1)(conv)
	    #conv = SpatialDropout2D(rate = dropout)(conv)
	    
	    conv = Conv2D(filters = filters* 2, kernel_size = (3,3), strides = (1,1), padding = "same", dilation_rate = (1,1),
	                  activation = "relu")(conv) ; print("conv ", conv.shape)
	    conv = BatchNormalization(axis=-1)(conv)
	    #conv = SpatialDropout2D(rate = dropout)(conv)
	    
	    conv = Conv2D(filters = filters* 3, kernel_size = (3,3), strides = (1,1), padding = "same", dilation_rate = (1,1),
	                  activation = "relu")(conv) ; print("conv ", conv.shape)
	    conv = BatchNormalization(axis=-1)(conv)
	    #conv = SpatialDropout2D(rate = dropout)(conv)
	    
	    conv = Conv2D(filters = filters* 3, kernel_size = (5,5), strides = (1,1), padding = "same", dilation_rate = (1,1),
	                  activation = "relu")(conv) ; print("conv ", conv.shape)
	    conv = BatchNormalization(axis=-1)(conv)
	    #conv = SpatialDropout2D(rate = dropout)(conv)
	    
	    conv = Conv2D(filters = filters* 3, kernel_size = (5,5), strides = (1,1), padding = "same", dilation_rate = (1,1),
	                  activation = "relu")(conv) ; print("conv ", conv.shape)
	    conv = BatchNormalization(axis=-1)(conv)
	    #conv = SpatialDropout2D(rate = dropout)(conv)
	    
	    conv = Conv2D(filters = filters* 3, kernel_size = (5,5), strides = (1,1), padding = "same", dilation_rate = (1,1),
	                  activation = "relu")(conv) ; print("conv ", conv.shape)
	    conv = BatchNormalization(axis=-1)(conv)
	    #conv = SpatialDropout2D(rate = dropout)(conv)
	    
	    return conv

	def FullModel(self):

		ip = Input(shape = (audio_ip_shape[0], audio_ip_shape[1], 2)) ; print("input_audio", ip.shape) 
		ip_embeddings_1 = Input(shape = (video_ip_shape[0], video_ip_shape[1])); print("ip video", ip_embeddings_1.shape)  #[75, 512]
		ip_embeddings_2 = Input(shape = (video_ip_shape[0], video_ip_shape[1])); print("ip video", ip_embeddings_2.shape)  #[75, 512]

		ip_magnitude = Lambda(lambda x : x[:,:,:,0],name="ip_mag")(ip); print("ip_mag ", ip_magnitude.shape)  #takes magnitude from stack[magnitude,phase]
		ip_phase = Lambda(lambda x : tf.expand_dims(x[:,:,:,1], axis = -1),name="ip_phase")(ip); print("ip_phase ", ip_phase.shape)  #takes phase from stack[magnitude,phase]

		ip_embeddings_1_expanded = Lambda(lambda x : tf.expand_dims(x, axis = -1))(ip_embeddings_1)
		ip_embeddings_2_expanded = Lambda(lambda x : tf.expand_dims(x, axis = -1))(ip_embeddings_2)

		audio_stream = AudioModel(ip)

		stream_1 = self.conv1(ip_embeddings_1)
		stream_1 = self.bn1(stream_1)
		stream_1 = self.conv2(stream_1)
		stream_1 = self.bn2(stream_1)
		stream_1 = self.conv3(stream_1)
		stream_1 = self.bn3(stream_1)
		stream_1 = self.conv4(stream_1)
		stream_1 = self.bn4(stream_1)
		stream_1 = self.conv5(stream_1)
		stream_1 = self.bn5(stream_1)
		stream_1 = self.conv6(stream_1)
		stream_1 = self.bn6(stream_1)
		stream_1 = self.conv7(stream_1) 
		video_stream_1 = self.conv8(stream_1)

		stream_2 = self.conv1(ip_embeddings_2)
		stream_2 = self.bn1(stream_2)
		stream_2 = self.conv2(stream_2)
		stream_2 = self.bn2(stream_2)
		stream_2 = self.conv3(stream_2)
		stream_2 = self.bn3(stream_2)
		stream_2 = self.conv4(stream_2)
		stream_2 = self.bn4(stream_2)
		stream_2 = self.conv5(stream_2)
		stream_2 = self.bn5(stream_2)
		stream_2 = self.conv6(stream_2)
		stream_2 = self.bn6(stream_2)
		stream_2 = self.conv7(stream_2)
		video_stream_2 = self.conv8(stream_2)

		audio_flatten = TimeDistributed(Flatten())(audio_stream) 
		video_flatten_1 = TimeDistributed(Flatten())(video_stream_1) 
		video_flatten_2 = TimeDistributed(Flatten())(video_stream_2)

		print("video Streams ", video_stream_1.shape, video_stream_2.shape)
		print("Flatten Streams", video_flatten_1.shape, video_flatten_2.shape, audio_flatten.shape)

		concated = concatenate([audio_flatten, video_flatten_1, video_flatten_2], axis = 2) ;print("concat shape ", concated.shape)

		lstm = Bidirectional(LSTM(units = 64, return_sequences = True, activation = "tanh"))(concated)   ;print("lstm", lstm.shape)

		flatten = Flatten()(lstm) ;print("flatten ", flatten.shape)

		dense = Dense(100, activation = "relu")(flatten)

		dense = Dense(2 * audio_ip_shape[0] * audio_ip_shape[1], activation = "sigmoid")(dense) ;print("dense final ",dense.shape)

		combo_mask = Reshape([2 , audio_ip_shape[0], audio_ip_shape[1]])(dense) ; print("combo_mask ", combo_mask.shape)
		mask_1 = Lambda(lambda x : x[:,0])(combo_mask) ;print("mask 1 ", mask_1.shape)
		mask_2 = Lambda(lambda x : x[:,1])(combo_mask) ;print("mask 2 ", mask_2.shape)

		output_mag_1 = Lambda(lambda x : tf.multiply(x[0], x[1]), name = "mask_multiply_1")([ip_magnitude, mask_1]) ; print("output_mag_1", output_mag_1.shape)
		output_mag_2 = Lambda(lambda x : tf.multiply(x[0], x[1]), name = "mask_multiply_2")([ip_magnitude, mask_2]) ; print("output_mag_2", output_mag_2.shape)

		output_mag_1 = Lambda(lambda x : tf.expand_dims(x, axis= -1), name= "expand_dim_1")(output_mag_1) ; print("output_mag_expand_1", output_mag_1.shape)
		output_mag_2 = Lambda(lambda x : tf.expand_dims(x, axis= -1), name= "expand_dim_2")(output_mag_2) ; print("output_mag_expand_2", output_mag_2.shape)

		output_final_1 = Lambda(lambda x : tf.concat(values=[x[0], x[1]], axis = -1),name="concat_mag_phase_1")([output_mag_1, ip_phase]) ; print("output_final_1 ", output_final_1.shape)
		output_final_2 = Lambda(lambda x : tf.concat(values=[x[0], x[1]], axis = -1),name="concat_mag_phase_2")([output_mag_2, ip_phase]) ; print("output_final_2 ", output_final_2.shape)

		model = Model([ip, ip_embeddings_1, ip_embeddings_2], [output_final_1, output_final_2])

		return model































