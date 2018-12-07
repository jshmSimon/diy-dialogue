"""
Generate samples.txt from dialogue_text.txt
And display descriptive statistics
"""

import collections
import nltk

# extract data from filename into samples.txt
# every line in samples.txt is a conservation that contains at least 4 turns
def read_data(filename):
	ret = []
	out_file = open("samples.txt", "w", encoding='utf-8')
	with open(filename, "r", encoding='utf-8') as f:
		for line in f:
			line = line.replace("\r", "").replace("__eou__\n", "").lower().strip()
			split = line.split("__eou__")

			while "" in split:
				split.remove("")

			# only conservation containing at least 4 turns will be saved into samples.txt
			if len(split) >=4:
				ret.append(line)
				out_file.write(line + "\n")

	out_file.close()
	return ret


lines = read_data("dialogues_text.txt")

# display descriptive statistics
def count(lines):
	num_conv = 0 # count total conservations
	num_turn = 0 # count total turns
	num_utter = 0 # count total utterances
	num_words = 0 # count total words
	words = []
	for line in lines:
		num_conv += 1
		turns = line.split("__eou__")
		num_turn += len(turns)

		for turn in turns:
			sents = nltk.sent_tokenize(turn)
			num_utter += len(sents)
			for sent in sents:
				li = nltk.word_tokenize(sent)
				num_words += len(li)
				words += li


	print(" # num_conv: %d" % num_conv)
	print(" # num_turn/num_conv: {}".format(num_turn/num_conv))
	print(" # num_utter/num_conv: {}".format(num_utter/num_conv))
	print(" # num_words/num_utter: {}".format(num_words/num_utter))

	"""
	The following will be executed in process_vocab.py
	"""
	# counter = collections.Counter(words)
	# freq = counter.most_common()
	# f = open("vocab.txt", "w")
	# for (w,i) in freq:
	# 	f.write(w + "\n")
	# f.close()
	# print()
	# print(" # vocab_size: %d" % len(freq))


count(lines)



 # num_conv: 12553
 # num_turn/num_conv: 9.113598343025572
 # num_utter/num_conv: 13.089062375527762
 # num_words/num_utter: 8.673903120378316

 # vocab_size: 22967
