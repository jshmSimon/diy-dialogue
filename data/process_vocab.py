"""
Generate vocab.txt from train.txt and test.txt
vocab.txt contains words appear in train.txt and test.txt
Words in vocab.txt are unique and descending sorted by frequency
"""

import collections
from nltk.tokenize import RegexpTokenizer

# create a tokenizer that includes words and excludes punctuations
tokenizer = RegexpTokenizer(r'\w+')

def main():
	# word_list is to contain all words in train.txt and test.txt
	word_list = []

	# extract words from train.txt
	with open("train.txt", "r", encoding='utf-8') as f:
		for line in f:
			line = line.replace("\r", "").replace("\n", "").replace(" __eou__ ", " ").lower().strip()
			new = tokenizer.tokenize(line)
			word_list += new
	
	# extract words from test.txt
	with open("test.txt", "r", encoding='utf-8') as f:
		for line in f:
			line = line.replace("\r", "").replace("\n", "").replace(" __eou__ ", " ").lower().strip()
			new = tokenizer.tokenize(line)
			word_list += new

	# count words in word_list
	counter = collections.Counter(word_list)

	# words are descending sorted by frequency
	word_freq = counter.most_common()

	print(word_freq)
	print(len(word_freq))

	# save vocab as vocab.txt
	# note that vocab.txt only contains words, except frequency
	out_file = open("vocab.txt", "w")
	for (w, i) in word_freq:
		out_file.write(w + "\n")
	out_file.close()

if __name__ == '__main__':
	main()