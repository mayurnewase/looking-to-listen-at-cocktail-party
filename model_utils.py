



def trainGen(batch_size, train_dic, num_steps):
    #for i in range(3):  # for epochs
    keys = list(train_dic.keys())
    
    while 1:
        for step in range (num_steps):  #1 batch
            x_audio = []
            x_video_1 = []
            x_video_2 = []   
            
            y_audio_1 = []
            y_audio_2 = []   

            batch_keys = keys[step* batch_size : (step+ 1)* batch_size]
            #clean_audio_1, clean_audio_2, mixed_audio, video_1, video_2
            
            for key in batch_keys:
                x_audio.append(pickle.load(open(train_dic[key][2], "rb")))  #ip_audio
        
                x_video_1.append(pickle.load(open(train_dic[key][3], "rb")))  #video
                x_video_2.append(pickle.load(open(train_dic[key][4], "rb")))  #video
            
                y_audio_1.append(pickle.load(open(train_dic[key][0], "rb")))   #audio only
                y_audio_2.append(pickle.load(open(train_dic[key][1], "rb")))   #audio only

            x_audio_arr = np.array(x_audio)
            x_video_arr_1 = np.array(x_video_1)
            x_video_arr_2 = np.array(x_video_2)
            y_audio_arr_1 = np.array(y_audio_1)
            y_audio_arr_2 = np.array(y_audio_2)
            
            yield [x_audio_arr, x_video_arr_1, x_video_arr_2], [y_audio_arr_1, y_audio_arr_2]

def valGen(batch_size, val_dic, num_steps):
    #for i in range(3):  # for epochs
    keys = list(val_dic.keys())
    
    while 1:
        for step in range (num_steps):  #1 batch
            x_audio = []
            x_video_1 = []
            x_video_2 = []   
            
            y_audio_1 = []
            y_audio_2 = []   

            batch_keys = keys[step* batch_size : (step+ 1)* batch_size]
            #clean_audio_1, clean_audio_2, mixed_audio, video_1, video_2
            
            for key in batch_keys:
                x_audio.append(pickle.load(open(val_dic[key][2], "rb")))  #ip_audio
        
                x_video_1.append(pickle.load(open(val_dic[key][3], "rb")))  #video
                x_video_2.append(pickle.load(open(val_dic[key][4], "rb")))  #video
            
                y_audio_1.append(pickle.load(open(val_dic[key][0], "rb")))   #audio only
                y_audio_2.append(pickle.load(open(val_dic[key][1], "rb")))   #audio only

            x_audio_arr = np.array(x_audio)
            x_video_arr_1 = np.array(x_video_1)
            x_video_arr_2 = np.array(x_video_2)
            y_audio_arr_1 = np.array(y_audio_1)
            y_audio_arr_2 = np.array(y_audio_2)
            
            yield [x_audio_arr, x_video_arr_1, x_video_arr_2], [y_audio_arr_1, y_audio_arr_2]


def loss(spect1, spect2):
    loss = tf.sqrt(tf.nn.l2_loss(spect1[:,:,:,0] - spect2[:,:,:,0]))
    #loss = tf.sqrt(tf.nn.l2_loss(spect1 - spect2))
    return loss

    