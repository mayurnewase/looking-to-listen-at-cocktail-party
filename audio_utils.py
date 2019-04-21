import tensorflow as tf
import functools
from tensorflow.contrib.framework.python.ops import audio_ops
from tensorflow.python.ops import io_ops
import tensorflow_probability as tfp


def snr(signal_mag, noise_mag):
  
  signal_power = np.sum(signal_mag ** 2)
  noisy_power = np.sum(noise_mag ** 2)
  snr_power = signal_power/noisy_power #;print(str(snr) + " of 1")  #best 1

  signal_power_db = 10 * np.log10(np.sum(signal_mag ** 2))
  noisy_power_db = 10 * np.log10(np.sum(noise_mag ** 2))
  snr_db = signal_power_db - noisy_power_db #;print(str(snr_db) + " db of 0") #best 0
  return snr_power, snr_db


def griffin_lim(magnitude, n_fft, hop_length, n_iterations):
	phase_angle = np.pi * np.random.rand(*magnitude.shape)
	D = invert_magnitude_phase(magnitude, phase_angle)
	signal = librosa.istft(D, hop_length=hop_length)

	for i in range(n_iterations):
		D = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)
		_, phase = librosa.magphase(D)
		phase_angle = np.angle(phase)

		D = invert_magnitude_phase(magnitude, phase_angle)
		signal = librosa.istft(D, hop_length=hop_length)

	return signal

def invert_magnitude_phase(magnitude, phase_angle):
	phase = np.cos(phase_angle) + 1.j * np.sin(phase_angle)
	return magnitude * phase

def tf_istft(stft):
    with tf.Session(graph=tf.Graph()) as sess:
        inv_st = tf.signal.inverse_stft(stft, frame_length = frame_length, frame_step = frame_step, fft_length=fft_length, window_fn=tf.signal.hann_window)
        inv_st_op = sess.run(inv_st)
        return inv_st_op.T

def write_file(data, name, rate = sample_freq):
    encoded_audio_data = tf.contrib.ffmpeg.encode_audio(
        data, file_format="wav", samples_per_second= rate)

    write_file_op = tf.write_file(name, encoded_audio_data)

    with tf.Session() as sess:
        sess.run(write_file_op)
        
def get_complex(stft, power_compression = power_compression):
    if power_compression:
      stft[:,:,0] = stft[:,:,0] ** (1/0.3)
      
    x = np.zeros((stft.shape[0], stft.shape[1]), dtype = "complex64")
    for i in range(0, len(stft[:,0,0])):
        j=0
        for j in range(0, len(stft[0,:,0])):
            x[i, j] = np.complex(stft[i,j,0] * np.cos(stft[i,j,1]) ,stft[i,j,0] * np.sin(stft[i,j,1]))
        
    return x

def tf_stft(audio, power_compression = power_compression):
      
    with tf.Session(graph=tf.Graph()) as sess:
        holder = tf.placeholder(tf.string, [])
        file = tf.read_file(holder)
        #decoder = audio_ops.decode_wav(file, desired_channels = 1, desired_samples = sample_freq* duration)
        decoder = tf.contrib.ffmpeg.decode_audio(file, file_format = "wav", samples_per_second = sample_freq, channel_count = 1)
        stft = tf.signal.stft(tf.transpose(decoder), frame_length = frame_length, frame_step = frame_step, fft_length = fft_length, 
                              window_fn= tf.signal.hann_window)
        
        amp = tf.squeeze(tf.abs(stft))
        phase = tf.squeeze(tf.angle(stft))
        if power_compression:
          amp = amp ** 0.3
        
        stacked = tf.stack([amp, phase], 2)
        
        op = sess.run([decoder, stft, amp, phase, stacked], feed_dict = {holder: audio})
        #op = sess.run([decoder, stft, amp, phase, stacked, mel_mat, mel_spec, amp_rec], feed_dict = {holder: audio})
                                      
    return op    