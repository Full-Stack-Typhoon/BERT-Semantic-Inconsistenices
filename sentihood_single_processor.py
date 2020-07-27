import csv
import os
import numpy
import pandas as pd



sentihood_single_data = "/home/vp1274/ABSA-BERT-pair/data/sentihood/bert-single/"
sentihood_datasets1 = ["loc1_general","loc1_safety","loc1_transit","loc1_price"]
sentihood_output1 = "/home/vp1274/ABSA-BERT-pair/data/sentihood/bert-single/loc1/"
sentihood_single_data = "/home/vp1274/ABSA-BERT-pair/data/sentihood/bert-single/"
sentihood_datasets2 = ["loc2_general","loc2_safety","loc2_transit","loc2_price"]
sentihood_output2 = "/home/vp1274/ABSA-BERT-pair/data/sentihood/bert-single/loc2/"
semeval_single_data = "/home/vp1274/ABSA-BERT-pair/data/semeval2014/bert-single/"
semeval_datasets = ["ambience","anecdotes","price","food","service"]
semeval_output = "/home/vp1274/ABSA-BERT-pair/data/semeval2014/bert-single/combined/"
#tasks = ["train","dev","test"]
tasks =["train","test"]

def process(base_dir, datasets, output):
	for task in tasks:
		data = [] 
		for dataset in datasets:
			data.append(pd.read_csv((os.path.join(base_dir + dataset,task +  ".csv")),header=None,sep="\t").values)
		out_data = []
		for cnt, d in enumerate(data):
			if cnt==0:
				for line in d:
					out_line = line
					out_line[-1] = datasets[cnt].split("_")[1]+  "-" + line[-1]
					out_data.append(out_line)
			else:
				for line_cnt, line in enumerate(d):
					#print(len(out_data[line_cnt]))
					out_data[line_cnt] = numpy.append(out_data[line_cnt],(datasets[cnt].split("_")[1]+ "-" + line[-1]))
		out_file = open(output+task+".csv","w",encoding="utf-8")
		for line in out_data:
			line = [str(i) for i in line]
			out_line = "\t".join(line)+"\n"
			out_file.write(out_line)
		out_file.close()
		#return	

#process(sentihood_single_data, sentihood_datasets1,sentihood_output1)
#process(sentihood_single_data, sentihood_datasets2,sentihood_output2)
process(semeval_single_data, semeval_datasets,semeval_output)
