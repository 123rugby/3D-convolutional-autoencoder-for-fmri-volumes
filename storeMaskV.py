import os
import numpy as np
import nibabel as nib
import tensorflow as tf
import urllib2
import pandas as pd

#access already shuffled file names
randomNameR = open('randomNames.txt', 'r')
names = randomNameR.read().split('\n')
del names[1035]

#some functions for tfrecords
def _bytes_feature(value):
	return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
def _int64_feature(value):
	return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

# string variables needed to access tfrecords files
tfrecordsPath = "/tfrecords/"
tfrecordsExt = "MC.tfrecords"

# string variables needed to access fmri files
path = "/abide/Preprocessed_data/"
ext = "_func_minimal.nii.gz"

#string variables needed to access mask files
maskUrl = "https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative/Outputs/cpac/filt_global/func_mask/"
maskext = "_func_mask.nii.gz"

#start and end points of the for loop below. Needed to access all 1035 fmri files, and put into 15 tfrecords files, each with 69 fmri files.
start = 1
end = 70

# keep track of number of volumes in each tfrecord file.
volPerFile = open("volPerFile.txt", 'wb')

print "\nStarting conversion and masking...\n"

#for each tfrecords file
for fileNo in range(15):

	#create a tfrecords file
	tfrecords_filename = tfrecordsPath + str(fileNo + 1) + tfrecordsExt
	
	#create a writer to write in it.
	writer = tf.python_io.TFRecordWriter(tfrecords_filename)
	
	#keep count of volumes
	vol_counter = 0
	
	#for every 69 files that go into the above tf file
	for i in range(start, end):
	
		#loading the image
		fil = str(names[i])
		file_name = path + fil + ext
		img = nib.load(file_name)
		img_data = img.get_data()
		
		#loading the mask
		url = maskUrl + fil + maskext
		u = urllib2.urlopen(url)
		f = open(maskext, 'wb')
		meta = u.info()
		file_size = int(meta.getheaders("Content-Length")[0])
		#print "Downloading mask for : %s Bytes: %s" % (fil, file_size)
		file_size_dl = 0
		block_sz = 8192
		while True:
			buffer = u.read(block_sz)
			if not buffer:
				break
			file_size_dl += len(buffer)
			f.write(buffer)
			status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
			status = status + chr(8)*(len(status)+1)
			#print status,
		f.close()
		#print '\n'	
		mask = nib.load(maskext)
		mask_data = mask.get_data()
		mask_data32 = mask_data.astype(np.float32)
		
		# some stuff needed for next loop.
		img_shape = img_data.shape
		img_shape = np.asarray(img_shape)
		W = img_shape.item(3)
		I32 = img_data.astype(np.float32)
		
		# for each volume in a file
		for vol in range(W):
			volume = I32[:,:,:,vol]
			
			# mask the volumes
			volume = np.multiply(volume, mask_data32)
			
			#crop 1 from each dimension
			volume = np.delete(volume, 60, 0)
			volume = np.delete(volume, 60, 2)
			volume = np.delete(volume, 72, 1)
			
			#flatten and write to tf file
			vol_str = volume.tostring()
			example = tf.train.Example(features=tf.train.Features(feature={'vol_raw': _bytes_feature(vol_str)}))
			writer.write(example.SerializeToString())
			
			#keep track of number of volumes added to the tf file.
			vol_counter = vol_counter + 1
			
		#print 'file\t' + str(i) + '\tmasked and added to tfrecords file'
		
	print vol_counter
	
	writer.close()
	
	# move to next 69 files to be added to the next tf records file.
	start = start + 69
	end = start + 69
	
	#keep track of number of volumes
	volPerFile.write(str(vol_counter) + '\n')

#close the file writer
volPerFile.close()
	

