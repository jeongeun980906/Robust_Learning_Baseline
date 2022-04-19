cd ..
python noise-adaptation.py --noise_type instance --noise_rate 0.2 --dataset cifar10
python noise-adaptation.py --noise_type instance --noise_rate 0.4 --dataset cifar10

python f-correction.py --noise_rate 0.2 --dataset cifar10 --epochs 200 --noise_type instance
python f-correction.py --noise_rate 0.4 --dataset cifar10 --epochs 200 --noise_type instance

python Co-teaching.py --noise_rate 0.2 --dataset cifar10 --model_type coteaching --noise_type instance --lr 1e-3
python Co-teaching.py --noise_rate 0.4 --dataset cifar10 --model_type coteaching --noise_type instance  --lr 1e-3

python Co-teaching.py --noise_rate 0.2 --dataset cifar10 --noise_type instance --lr 1e-3
python Co-teaching.py --noise_rate 0.4 --dataset cifar10 --noise_type instance  --lr 1e-3

python JoCoR.py --noise_rate 0.2 --noise_type instance --n_epoch 200 --dataset cifar10 --lr 1e-3
python JoCoR.py --noise_rate 0.4 --noise_type instance --n_epoch 200 --dataset cifar10 --lr 1e-3

python Dividemix.py --num_epochs 200 --dataset cifar10 --noise_mode instance --r 0.2 --lr_step 100 --Adam --lr 1e-3
python Dividemix.py --num_epochs 200 --dataset cifar10 --noise_mode instance --r 0.4 --lr_step 100 --Adam --lr 1e-3
