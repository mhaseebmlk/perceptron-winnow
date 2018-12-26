import sys
import random

import numpy as np

from parser import Parser
from f_score import macro_f1

class AveragePerceptron:
	def __init__(self,training_set,training_set_sz,num_epochs,learning_rate):

		self.training_data=training_set[:training_set_sz]  #get only the required amount from the training set

		# shuffle the training data
		random.shuffle(self.training_data)

		self.num_epochs=num_epochs
		self.learning_rate=learning_rate
		self.training_set_sz=training_set_sz

		self.num_features=785 # 28x28 + 1 for the bias
		self.num_digits=10 # this will be the number of perceptrons

		self.perceptrons=[np.zeros(self.num_features) for i in range(self.num_digits)]

		# # initializing perceptrons with random weights
		# self.perceptrons=[np.random.rand(self.num_features) for i in range(self.num_digits)]

		# initializing perceptron averages to zeros
		self.perceptron_avgs=[np.zeros(self.num_features) for i in range(self.num_digits)]

	def train(self):
		for i in range(len(self.perceptrons)):
			digit=i
			self.perceptron_train(digit)

	def perceptron_train(self,digit):
		# c=1.0
		for _ in range(self.num_epochs):
			for img, lbl in self.training_data:
				y=1 if digit==lbl else -1

				# compute activation for this example and this perceptron
				act=np.dot(self.perceptrons[digit],img)

				# update weight
				if y*act <= 0:
					self.perceptrons[digit]+= self.learning_rate*y*img

				self.perceptron_avgs[digit]+=self.perceptrons[digit]

				# c+=1.0
		# self.perceptron_avgs[digit]	= self.perceptrons[digit] - (self.perceptron_avgs[digit]/float(c))

	def perceptron_test(self,img):
		acts=[np.dot(self.perceptron_avgs[digit],img) for digit in range(len(self.perceptron_avgs))]
		predicted_digit=acts.index(max(acts))
		return predicted_digit

def print_usage():
	print('Usage:')
	print('python average_perceptron.py [size of training set] [no of epochs] [learning rate] [path to DATA_FOLDER]')

def check_cmd_line_args():
	if len(sys.argv) != 5:
		print_usage()
		sys.exit()

def read_cmd_line_args():
	return int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3]),sys.argv[4]

def main():
	check_cmd_line_args()
	training_set_sz,num_epochs,learning_rate,data_folder=read_cmd_line_args()

	train_data=data_folder+'/'+'train-images-idx3-ubyte.gz'
	train_label_data=data_folder+'/'+'train-labels-idx1-ubyte.gz'
	training_parser=Parser(train_data,train_label_data)
	training_parsed=training_parser.parse()

	test_data=data_folder+'/'+'t10k-images-idx3-ubyte.gz'
	test_label_data=data_folder+'/'+'t10k-labels-idx1-ubyte.gz'
	testing_parser=Parser(test_data,test_label_data)
	testing_parsed=testing_parser.parse()

	# hyper-parameter tuning:
	# TRAINING_SET SIZE
	sizes=[500, 750, 1000, 1250, 1500, 1750, 2000, 2250, 2500, 2750, 3000, 3250, 3500, 3750, 4000, 4250, 4500, 4750, 5000, 5250, 5500, 5750, 6000, 6250, 6500, 6750, 7000, 7250, 7500, 7750, 8000, 8250, 8500, 8750, 9000, 9250, 9500, 9750, 10000]
	sz_tuning_params=(sizes,50,0.001)

	# NUM EPOCHS
	epochs=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
	epoch_tuning_params=(10000,epochs,0.001)

	# LEARNING RATE
	rates=[0.0001, 0.001, 0.01, 0.1]
	rate_tuning_params=(10000,50,rates)

	# for sz in sz_tuning_params[0]:
	# 	training_set_sz=sz
	# 	num_epochs=sz_tuning_params[1]
	# 	learning_rate=sz_tuning_params[2]

	average_perceptron=AveragePerceptron(training_parsed,training_set_sz,num_epochs,learning_rate)

	# Training
	average_perceptron.train()

	# Training F1 Score and accuracy
	training_set=training_parsed[:training_set_sz]
	training_accuracy=0.0
	true_labels=[]
	predicted_labels=[]
	for img, lbl in training_set:
		predicted_digit=average_perceptron.perceptron_test(img)
		true_labels.append(lbl)
		predicted_labels.append(predicted_digit)
		if predicted_digit==lbl:
			training_accuracy+=1
	training_f1=macro_f1(true_labels,predicted_labels) * 100
	training_accuracy=(training_accuracy/len(testing_parsed)) * 100

	# Test F1 score and accruacy
	test_accuracy=0.0
	true_labels=[]
	predicted_labels=[]
	for img, lbl in testing_parsed:
		predicted_digit=average_perceptron.perceptron_test(img)
		true_labels.append(lbl)
		predicted_labels.append(predicted_digit)
		if predicted_digit==lbl:
			test_accuracy+=1
	test_f1=macro_f1(true_labels,predicted_labels) * 100
	test_accuracy=(test_accuracy/len(testing_parsed)) * 100

		# print '{},{},{},{},{}'.format(sz,test_accuracy,test_f1,training_accuracy,training_f1)

	print 'Training F1 score:',training_f1
	print 'Test F1 score:',test_f1

if __name__ == '__main__':
	main()
