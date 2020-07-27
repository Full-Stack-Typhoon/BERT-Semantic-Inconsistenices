import os

from data_utils_sentihood import *

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("context_window", help="context_window",
                    type=int)
args = parser.parse_args()
context_size = args.context_window

data_dir='../data/sentihood/'
aspect2idx = {
    'general': 0,
    'price': 1,
    'transit-location': 2,
    'safety': 3,
}

(train, train_aspect_idx), (val, val_aspect_idx), (test, test_aspect_idx) = load_task(data_dir, aspect2idx)

print("len(train) = ", len(train))
print("len(val) = ", len(val))
print("len(test) = ", len(test))

train.sort(key=lambda x:x[2]+str(x[0])+x[3][0])
val.sort(key=lambda x:x[2]+str(x[0])+x[3][0])
test.sort(key=lambda x:x[2]+str(x[0])+x[3][0])

dir_path = data_dir+'bert-pair/'
if not os.path.exists(dir_path):
    os.makedirs(dir_path)

sentiments=["None","Positive","Negative"]
sub_dir = "{}/".format(context_size)
def write_NLI_B(filename, data):
    with open(dir_path+sub_dir+filename,"w",encoding="utf-8") as f:
        f.write("id\tsentence1\tsentence2\tlabel\n")
        for v in data:
            for sentiment in sentiments:
                f.write(str(v[0])+"\t")
                word=v[1][0].lower()
                if word=='location1':f.write('location - 1')
                elif word=='location2':f.write('location - 2')
                elif word[0]=='\'':f.write("\' "+word[1:])
                else:f.write(word)
                loc_index = 0
                for i in range(1,len(v[1])):
                    word=v[1][i].lower()
                    f.write(" ")
                    if word == 'location1':
                        f.write('location - 1')
                        if v[2] == 'LOCATION1':
                            loc_index = i
                    elif word == 'location2':
                        f.write('location - 2')
                        if v[2] == 'LOCATION2':
                            loc_index = i
                    elif word[0] == '\'':
                        f.write("\' " + word[1:])
                    else:
                        f.write(word)
                f.write("\t")
                f.write(sentiment+" - ")
                if len(v[3])==1:    
                    f.write(v[3][0]+" - ")
                else:
                    f.write("transit location - ")
                for d in range(-1*context_size, context_size+1):
                    if d != 0 and loc_index + d >= 0 and loc_index + d < len(v[1]):
                        f.write(v[1][loc_index + d])
                        f.write(" ")
                    elif d == 0:
                        if v[2]=='LOCATION1':f.write('location - 1 ')
                        if v[2]=='LOCATION2':f.write('location - 2 ')
                f.write("\t")
                if v[4]==sentiment:
                    f.write("1\n")
                else:
                    f.write("0\n")  
  

write_NLI_B("train_NLI_B.tsv", train)
write_NLI_B("dev_NLI_B.tsv", val)
write_NLI_B("test_NLI_B.tsv", test)
    

def write_QA_B(filename, data):            
    with open(dir_path+sub_dir+filename,"w",encoding="utf-8") as f:
        f.write("id\tsentence1\tsentence2\tlabel\n")
        for v in data:
            for sentiment in sentiments:
                f.write(str(v[0])+"\t")
                word=v[1][0].lower()
                if word=='location1':f.write('location - 1')
                elif word=='location2':f.write('location - 2')
                elif word[0]=='\'':f.write("\' "+word[1:])
                else:f.write(word)
                loc_index = 0
                for i in range(1,len(v[1])):
                    word=v[1][i].lower()
                    f.write(" ")
                    if word == 'location1':
                        f.write('location - 1')
                        if v[2] == 'LOCATION1':
                            loc_index = i
                    elif word == 'location2':
                        f.write('location - 2')
                        if v[2] == 'LOCATION2':
                            loc_index = i
                    elif word[0] == '\'':
                        f.write("\' " + word[1:])
                    else:
                        f.write(word)
                f.write("\t")
                f.write("the polarity of the aspect ")
                if len(v[3])==1:
                    f.write(v[3][0] + " ")
                else:
                    f.write("transit location ")
                for d in range(-1*context_size, context_size+1):
                    if d != 0 and loc_index + d >= 0 and loc_index + d < len(v[1]):
                        f.write(v[1][loc_index + d])
                        f.write(" ")
                    elif d == 0:
                        if v[2]=='LOCATION1':f.write('of location - 1 - is ')
                        if v[2]=='LOCATION2':f.write('of location - 2 - is ')
                f.write(sentiment+" .\t")
                if v[4]==sentiment:
                    f.write("1\n")
                else:
                    f.write("0\n")

write_QA_B("train_QA_B.tsv", train)
write_QA_B("dev_QA_B.tsv", val)
write_QA_B("test_QA_B.tsv", test)
