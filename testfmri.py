
import os
import numpy as np
import tensorflow as tf
import math
import pandas as pd
import sys
from tensorflow.contrib.keras.python.keras import backend as K


# some important variables
IMAGE_HEIGHT = 60
IMAGE_WIDTH = 72
IMAGE_DEPTH = 60
batch_size = 32
volume_shape = [IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH]
input_shape = [batch_size, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_DEPTH, 1]
n_epochs = 100

# variable pertaining to the input pipeline
samples = 162000
num_threads= 4
min_after_dequeue = 4000
capacity = 15000 #15000

# tfrecords filenames with path, for the filename queue
tfrecords_filename1 = '/1MC.tfrecords'
tfrecords_filename2 = '/2MC.tfrecords'
tfrecords_filename3 = '/3MC.tfrecords'
tfrecords_filename4 = '/4MC.tfrecords'
tfrecords_filename5 = '/5MC.tfrecords'
tfrecords_filename6 = '/6MC.tfrecords'
tfrecords_filename7 = '/7MC.tfrecords'
tfrecords_filename8 = '/8MC.tfrecords'
tfrecords_filename9 = '/9MC.tfrecords'
tfrecords_filename10 = '/10MC.tfrecords'
tfrecords_filename11 = '/11MC.tfrecords'
tfrecords_filename12 = '/12MC.tfrecords'
tfrecords_filename13 = '/13MC.tfrecords'
tfrecords_filename14 = '/14MC.tfrecords'

# training related variables
n_batches = int((samples)/batch_size)
padding = 'SAME'
stride = [1,1,1]
learning_rate = 0.001
noise_factor = 0.3

# paths for saving summary and weights&biases.
logs_path = "summary/"
ws_path = "weights/"

# method to retrive a volume using the filename queue
def read_and_decode(filename_queue):
	reader = tf.TFRecordReader()
	key , serialized_example = reader.read(filename_queue)
	features = tf.parse_single_example(serialized_example, features={ 'vol_raw': tf.FixedLenFeature([], tf.string)})
	vol_str = tf.decode_raw(features['vol_raw'], tf.float32)
	volume = tf.reshape(vol_str, volume_shape)
	return volume

# method to sample a batch using the input pipeline
def input_pipeline():
	filename_queue = tf.train.string_input_producer([tfrecords_filename1, tfrecords_filename2, tfrecords_filename3, tfrecords_filename4, tfrecords_filename5, tfrecords_filename6, tfrecords_filename7, tfrecords_filename8, tfrecords_filename9, tfrecords_filename10, tfrecords_filename11, tfrecords_filename12],capacity = capacity, shuffle = True)
	volume = read_and_decode(filename_queue)
	volume_batch = tf.train.shuffle_batch([volume], batch_size=batch_size, capacity=capacity, num_threads= num_threads, min_after_dequeue=min_after_dequeue)
	finalbatch = tf.expand_dims(volume_batch, -1)
	
	return finalbatch

# method to sample a batch using the input pipeline from the validation data
def input_pipelineV():
	filename_queue = tf.train.string_input_producer([tfrecords_filename13, tfrecords_filename14], shuffle = True)
	volume = read_and_decode(filename_queue)
	volume_batch = tf.train.shuffle_batch([volume], batch_size=batch_size, capacity=capacity, num_threads= num_threads, min_after_dequeue=min_after_dequeue)
	finalbatch = tf.expand_dims(volume_batch, -1)
	
	return finalbatch

# the main method
def test_nii():
	
	# start of tensorflow graph
	 #input and target placeholders
	inputs_ = tf.placeholder(tf.float32, input_shape, name='inputs')
	targets_ = tf.placeholder(tf.float32, input_shape, name='targets')
	
	
	
	 #network 14
	  #encoder
	conv1 = tf.layers.conv3d(inputs= inputs_, filters=16, kernel_size=(3,3,3), padding= padding, strides = stride, activation=tf.nn.relu)	
	maxpool1 = tf.layers.max_pooling3d(conv1, pool_size=(2,2,2), strides=(2,2,2), padding= padding)
	conv2 = tf.layers.conv3d(inputs=maxpool1, filters=32, kernel_size=(3,3,3), padding= padding, strides = stride, activation=tf.nn.relu)
	maxpool2 = tf.layers.max_pooling3d(conv2, pool_size=(3,3,3), strides=(3,3,3), padding= padding)
	conv3 = tf.layers.conv3d(inputs=maxpool2, filters=96, kernel_size=(2,2,2), padding= padding , strides = stride, activation=tf.nn.relu)
	maxpool3 = tf.layers.max_pooling3d(conv3, pool_size=(2,2,2), strides=(2,2,2), padding= padding)
	  #latent internal representation

	  #decoder
	unpool1 = K.resize_volumes(maxpool3,2,2,2,"channels_last")
	deconv1 = tf.layers.conv3d_transpose(inputs=unpool1, filters=96, kernel_size=(2,2,2), padding= padding , strides = stride, activation=tf.nn.relu)
	unpool2 = K.resize_volumes(deconv1,3,3,3,"channels_last")
	deconv2 = tf.layers.conv3d_transpose(inputs=unpool2, filters=32, kernel_size=(3,3,3), padding= padding , strides = stride, activation=tf.nn.relu)
	unpool3 = K.resize_volumes(deconv2,2,2,2,"channels_last")
	deconv3 = tf.layers.conv3d_transpose(inputs=unpool3, filters=16, kernel_size=(3,3,3), padding= padding , strides = stride, activation=tf.nn.relu)
	
	output = tf.layers.dense(inputs=deconv3, units=1)
	output = tf.reshape(output, input_shape)
	  # output shape = input shape
	
	
	  #loss function, optimizer and a saver to save weights&biases
	loss = tf.divide(tf.norm(tf.subtract(targets_, output), ord = 'fro', axis = [1,2,3]), tf.norm(targets_, ord = 'fro', axis = [1,2,3]))
	cost = tf.reduce_mean(loss)
	opt = tf.train.AdamOptimizer(learning_rate).minimize(cost)
	all_saver = tf.train.Saver(max_to_keep = None)

	  #initializing a saver to save weights
	#enc_saver = tf.train.Saver({'conv1': conv1, 'conv1_1': conv1_1, 'maxpool1': maxpool1, 'conv2': conv2, 'maxpool2': maxpool2, 'conv3': conv3, 'maxpool3': maxpool3, 'conv4': conv4})
	  #initializing a restorer to restore weights
	#res_saver = tf.train.import_meta_graph('/weights/weights.meta')
	
	# summary nodes
	tf.summary.scalar("loss", loss)
	tf.summary.scalar("cost", cost)
	tf.summary.histogram("conv1",conv1)
	tf.summary.histogram("conv1_1",conv1_1)
	tf.summary.histogram("maxpool1",maxpool1)
	tf.summary.histogram("conv2",conv2)
	tf.summary.histogram("maxpool2",maxpool2)
	tf.summary.histogram("conv3",conv3)
	tf.summary.histogram("maxpool3",maxpool3)
	tf.summary.histogram("conv4",conv4)
	tf.summary.histogram("deconv4",deconv4)
	tf.summary.histogram("unpool3",unpool3)
	tf.summary.histogram("deconv3",deconv3)
	tf.summary.histogram("unpool2",unpool2)
	tf.summary.histogram("deconv2",deconv2)
	tf.summary.histogram("unpool1",unpool1)
	tf.summary.histogram("deconv1_1",deconv1_1)
	tf.summary.histogram("deconv1",deconv1)
	
	# summary operation and a writer to save it.
	summary_op = tf.summary.merge_all()
	writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
	

	# end of tensorflow graph
	
	# initializing tensorflow graph and a session
	init_op = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init_op)

	# making operation-variables to run our methods whenever needed during training
	fetch_op = input_pipeline()
	fetch_opV = input_pipelineV()

	# coordinator and queue runners to manage parallel sampling of batches from the input pipeline
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(sess=sess, coord=coord)
	
	# start of training
	counter = 0
	try:
		
		
		while not coord.should_stop():
			print '\nEpoch\t' + str(counter + 1) + '/' + str(n_epochs)
			for i in range(n_batches):
				#fetching a batch
				vol = sess.run(fetch_op)				
				nvol = np.asarray(vol)
				noisy_nvol = nvol + noise_factor * np.random.randn(*nvol.shape)
				batch_cost,_ = sess.run([cost,opt], feed_dict = {inputs_: noisy_nvol, targets_: nvol})
				if i%1000 == 0:
					print batch_cost					
				print '\r' + str(((i +1) * 100)/n_batches) + '%',
				sys.stdout.flush()
			counter = counter + 1			
			print("Epoch: {}/{}...".format(counter, n_epochs), "Training loss: {:.4f}".format(batch_cost))	
			#save weights and biases of the model
			all_saver.save(sess, ws_path + "model.ckpt", global_step = counter)
			#save weights and biases of the encoder
			#enc_saver.save(sess, ws_path + "enc.ckpt", global_step = counter)
			print 'Weights saved'
			#saving summary
			summary,_ = sess.run([summary_op,opt], feed_dict = {inputs_: nvol, targets_: nvol})
			writer.add_summary(summary, counter)
			print 'Summary saved'
			if counter >= n_epochs:
				break
		#checking validation error
		vol = sess.run(fetch_opV)
		nvol = np.asarray(vol)
		batch_cost,_ = sess.run([cost,opt], feed_dict = {inputs_: nvol, targets_: nvol})
		print 'Validation error' + str(batch_cost)
		
		
	except tf.errors.OutOfRangeError:
		print('Done training -- epoch limit reached')

	finally:
		coord.request_stop()
		
	coord.join(threads)
	sess.close()

	'''
	#code to restore weights
	with tf.Session() as sess:
		all_saver.restore(sess, "model.ckpt")
		print("Model restored.")
	'''
	
if __name__ == '__main__':
	test_nii()
