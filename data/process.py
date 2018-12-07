"""
Spilt samples.txt into trian.txt and test.txt
Split ratio is 0.5
"""

import os

def main():

	# read samples.txt
	all_sent = []
	file = os.path.join("samples.txt")
	with open(file, "r", encoding='utf-8') as f:
		for line in f:
			all_sent.append(line)

	# set split ratio
	split_ratio = 0.5
	data_size = len(all_sent)

	# determine train size according to data size and split ratio
	train_size = int(data_size * split_ratio)

	# split into train data and test data
	train_data = all_sent[:train_size]
	test_data = all_sent[train_size:]

	print(len(all_sent))
	print(len(train_data))
	print(len(test_data))

	# save train data as train.txt
	with open("train.txt", "w", encoding='utf-8') as f:
		for line in train_data:
			f.write(line)

	# save test data as test.txt
	with open("test.txt", "w", encoding='utf-8') as f:
		for line in test_data:
			f.write(line)

if __name__ == '__main__':
	main()