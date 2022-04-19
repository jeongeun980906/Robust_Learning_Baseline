cd ..
python noise-adaptation.py --noise_type instance --noise_rate 0.2 --dataset mnist
python noise-adaptation.py --noise_type instance --noise_rate 0.4 --dataset mnist

python f-correction.py --noise_rate 0.2 --dataset mnist --epochs 20 --noise_type instance
python f-correction.py --noise_rate 0.4 --dataset mnist --epochs 20 --noise_type instance

python Co-teaching.py --model_type coteaching --noise_rate 0.2 --dataset mnist --noise_type instance 
python Co-teaching.py --model_type coteaching --noise_rate 0.4 --dataset mnist --noise_type instance 

python Co-teaching.py --noise_rate 0.2 --dataset mnist --noise_type instance 
python Co-teaching.py --noise_rate 0.4 --dataset mnist --noise_type instance 

python JoCoR.py --noise_rate 0.2 --noise_type instance --n_epoch 20
python JoCoR.py --noise_rate 0.4 --noise_type instance --n_epoch 20

python Dividemix.py --num_epochs 20 --dataset mnist --noise_mode instance --r 0.2 --lr_step 10 --id 1
python Dividemix.py --num_epochs 20 --dataset mnist --noise_mode instance --r 0.4 --lr_step 10 --id 1