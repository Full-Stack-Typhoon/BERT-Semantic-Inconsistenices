# generate datasets

python generate_${1}_NLI_M.py ${2}
python generate_${1}_QA_M.py ${2}
python generate_${1}_NLI_B_QA_B.py ${2}
python generate_${1}_BERT_single.py ${2}
