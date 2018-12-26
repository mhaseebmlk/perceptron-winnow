"""
Module for parsing the MNIST data as required in the problem statement
"""
import struct
import random
import gzip

import numpy as np

class Parser:
	def __init__(self,data_file_path,label_file_path):
		"""
		:param data_file_path: The local path of the data file. The file must be in .gz format.
		:param label_file_path: The local path of the corresponding label file. The file must be in .gz format.

		It is the user's responsibility to pass in the correct pair of data and label files.

		:rtype: None
		"""
		self.data_file_path=data_file_path
		self.labels_file_path=label_file_path

	def parse(self,chunk_size=10000,shuffle=False):
		"""
		:param chunk_size: Returns chunk_size-many examples from the input data. Default value is None, meaning it will return the whole of the original data parsed.
		:param shuffle: If set to True, will shuffle the data before parsing. Default value is False.
		:rtype: array containing the parsed examples from the input data in the format [(example - a numpy array,lbl),...,(example,lbl)]
		"""

		unparsed_data = None
		parsed_data=[] 
		labels= None

		# Read the data 
		with gzip.open(self.data_file_path, 'rb') as f:
			zero, data_type, dims = struct.unpack('>HBB', f.read(4))
			shape = tuple(struct.unpack('>I', f.read(4))[0] \
	        	for d in range(dims))
			unparsed_data = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

		# Read the labels
		with gzip.open(self.labels_file_path, 'rb') as f:
			zero, data_type, dims = struct.unpack('>HBB', f.read(4))
			shape = tuple(struct.unpack('>I', f.read(4))[0] \
	        	for d in range(dims))
			labels = np.fromstring(f.read(), dtype=np.uint8).reshape(shape)

		combined_data=zip(unparsed_data,labels)

		if shuffle==True:
			random.Random(len(combined_data)).shuffle(combined_data)

		num_features_org=28 # the original image is 28x28
		num_features_parsed=785 # the new image vector will be 784+1

		for i in range(chunk_size):
			img=np.zeros(num_features_parsed)
			img[0]=1  # set the bias
			idx=1
			for j in range(num_features_org):
				for k in range(num_features_org):
					old_val=combined_data[i][0][j][k]
					new_val=int(round(old_val/255.0))
					img[idx]=new_val
					idx+=1

			parsed_data.append((img,combined_data[i][1]))

		return parsed_data