cd ..

python noise-adaptation.py --dataset clothing1m --batch_size 24
python Co-teaching.py --dataset clothing1m --lr 2e-3 --wd 1e-3
python Co-teaching.py --dataset clothing1m --model_type coteaching --lr 2e-3 --wd 1e-3
python JoCoR.py --dataset clothing1m 
python f-correction.py --dataset clothing1m --batch-size 32 --test-batch-size 32
python Dividemix.py --dataset clothing1m --gpu 7 --batch_size 32 --num_epochs 2 --data_path '/home/sungjoon.choi/seungyoun/Clothing1M' --num_class 14
