import json
from gensim.models import KeyedVectors
import numpy as np
from numpy import dot
from numpy.linalg import norm

def cos_sim(a,b):
	return dot(a, b)/(norm(a)*norm(b))

input_train_file =  "data/sentihood/sentihood-train.json"
output_train_file = "train.txt"

aspect_class = {"Positive" : "-pos","Negative" : "-neg"}
nightlife = ["club","clubs","restaurant","restaurants","pub","pubs","entertaining","entertainment","music","concert","dance","bar","bars","party","night","cinema","theatre","clubs/bars",""]
aspect_dict = {"nightlife": nightlife}

def read_data(input, output):
	
	wv_fp = "word2vec.6B.100d.txt"
	print("Loading word2vec file: {0}\n".format(wv_fp))
	# Load word vectors
	global wv
	wv = KeyedVectors.load_word2vec_format(wv_fp, binary=False)
	#print(wv)
	data = json.loads(open(input,'r').read())
	out_file = open(output,'w')
	i = 0
	for obj in data:
		if len(obj["opinions"])==0:
			continue
		sentiment = obj["opinions"][0]["sentiment"]
		aspect = obj["opinions"][0]["aspect"]
		if aspect.split("-")[0].lower() in wv:
			aspect_vec = wv[aspect.split("-")[0].lower()]
		else:
			aspect_vec = wv["unk"]
		#print(aspect_vec)
		target = obj["opinions"][0]["target_entity"]
		text = obj["text"].strip().split(" ")
		#print(text)
		"""for word in text:
                        if word.split("-")[0].lower() in wv:
                                word_vec = wv[word.lower()]
                        else:
                             	word_vec = wv["unk"]
                        if cos_sim(word_vec, aspect_vec) >= 0.7:
                                out_file.write(word + " " + aspect + aspect_class[sentiment] + "\n")
                        else:
                             	out_file.write(word + " O\n")"""
		for word in text:
			if word == target:
				out_file.write(word + " Target\n")
				continue

			if word.split("-")[0].lower() in wv:
                        	word_vec = wv[word.split("-")[0].lower()]
			else:
                     		word_vec = np.zeros(100)
			cos = cos_sim(word_vec, aspect_vec)
			print(word+" " + str(cos))
			if cos >= 0.65:
				out_file.write(word + " " + aspect + aspect_class[sentiment] + "\n")
			else:
				out_file.write(word + " O\n")
		out_file.write("\n")
	out_file.close()


read_data(input_train_file,  output_train_file)
		
		
