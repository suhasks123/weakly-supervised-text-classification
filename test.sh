
dataset=yelp
sup_source=labels
model=cnn
language_model=w2v

export CUDA_VISIBLE_DEVICES=0

python main.py --language_model ${language_model} --dataset ${dataset} --sup_source ${sup_source} --model ${model} --with_evaluation True
