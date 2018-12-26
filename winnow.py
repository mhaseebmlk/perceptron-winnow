import sys
import random

import numpy as np
import math

from parser import Parser
from f_score import macro_f1

class Winnow:
	def __init__(self,training_set,training_set_sz,num_epochs,learning_rate,theta):

		self.training_data=training_set[:training_set_sz]  #get only the required amount from the training set

		# shuffle the training data
		random.shuffle(self.training_data)

		self.num_epochs=num_epochs
		self.learning_rate=learning_rate
		self.training_set_sz=training_set_sz
		self.theta=theta

		self.num_features=785 # 28x28 + 1 for the bias
		self.num_digits=10 # this will be the number of perceptrons

		# initializing perceptrons with one weights
		self.perceptrons=[np.ones(self.num_features) for i in range(self.num_digits)]

		# initializing perceptron averages to ones
		self.perceptron_avgs=[np.ones(self.num_features) for i in range(self.num_digits)]

	def train(self):
		for i in range(len(self.perceptrons)):
			digit=i
			self.winnow_train(digit)

	def winnow_train(self,digit):
		for _ in range(self.num_epochs):
			for img, lbl in self.training_data:
				y=1 if digit==lbl else -1

				# compute activation for this example and this perceptron
				act=np.dot(self.perceptrons[digit],img)

				# if no mistake made, continue without changing
				if (y==1 and act>=self.theta) or (y==-1 and act<self.theta):
					self.perceptron_avgs[digit]+=self.perceptrons[digit]
					continue

				# if a mistake is made, update
				ones=np.where(img==1)
				for idx in ones:
					if y==1 and act<self.theta:
						self.perceptrons[digit][idx]=self.perceptrons[digit][idx]*self.learning_rate
					elif y==-1 and act>=self.theta:
						self.perceptrons[digit][idx]=self.perceptrons[digit][idx]*(1/self.learning_rate)

				self.perceptron_avgs[digit]+=self.perceptrons[digit]

	def winnow_test(self,img):
		acts=[np.dot(self.perceptron_avgs[digit],img) for digit in range(len(self.perceptron_avgs))]
		predicted_digit=acts.index(max(acts))
		return predicted_digit

def print_usage():
	print('Usage:')
	print('python winnow.py [size of training set] [no of epochs] [multiplication factor (>1)] [path to DATA_FOLDER]')

def check_cmd_line_args():
	if len(sys.argv) != 5 or float(sys.argv[3])<=1:
		print_usage()
		sys.exit()

def read_cmd_line_args():
	return int(sys.argv[1]),int(sys.argv[2]),float(sys.argv[3]),sys.argv[4]

def main():
	check_cmd_line_args()
	training_set_sz,num_epochs,mult_fact,data_folder=read_cmd_line_args()

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
	sz_tuning_params=(sizes,50,2.0)

	# NUM EPOCHS
	epochs=[10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
	epoch_tuning_params=(10000,epochs,2.0)

	# LEARNING RATE
	rates=[1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75, 4.0, 4.25, 4.5, 4.75, 5.0, 5.25, 5.5, 5.75, 6.0, 6.25, 6.5, 6.75, 7.0, 7.25, 7.5, 7.75, 8.0, 8.25, 8.5, 8.75, 9.0, 9.25, 9.5, 9.75, 10.0]
	rate_tuning_params=(10000,50,rates)

	# for rt in rate_tuning_params[2]:
	# 	training_set_sz=rate_tuning_params[0]
	# 	num_epochs=rate_tuning_params[1]
	# 	mult_fact=rt

	winnow=Winnow(training_parsed,training_set_sz,num_epochs,mult_fact,785)

	# Training
	winnow.train()

	# Training F1 Score and accuracy
	training_set=training_parsed[:training_set_sz]
	training_accuracy=0.0
	true_labels=[]
	predicted_labels=[]
	for img, lbl in training_set:
		predicted_digit=winnow.winnow_test(img)
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
		predicted_digit=winnow.winnow_test(img)
		true_labels.append(lbl)
		predicted_labels.append(predicted_digit)
		if predicted_digit==lbl:
			test_accuracy+=1
	test_f1=macro_f1(true_labels,predicted_labels) * 100
	test_accuracy=(test_accuracy/len(testing_parsed)) * 100

		# print '{},{},{},{},{}'.format(rt,test_accuracy,test_f1,training_accuracy,training_f1)

	print 'Training F1 score:',training_f1
	print 'Test F1 score:',test_f1

if __name__ == '__main__':
	main()