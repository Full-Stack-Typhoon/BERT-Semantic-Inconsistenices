CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=0 python run_classifier_TABSA.py --task_name sentihood_single --data_dir data/sentihood/bert-single/loc1/ --vocab_file uncased_L-12_H-768_A-12/vocab.txt --bert_config_file uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint uncased_L-12_H-768_A-12/pytorch_model.bin --eval_test --do_lower_case --max_seq_length 64 --train_batch_size 64 --learning_rate 2e-5 --num_train_epochs 4.0 --output_dir results/semeval/semeval_single_combined --seed 42
