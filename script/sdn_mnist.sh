cd ..

python noise-adaptation.py --noise_type symmetric --noise_rate 0.2 --dataset dirtymnist --lr 1e-4
python noise-adaptation.py --noise_type symmetric --noise_rate 0.5 --dataset dirtymnist --lr 1e-4
python noise-adaptation.py --noise_type symmetric --noise_rate 0.8 --dataset dirtymnist --lr 1e-4
python noise-adaptation.py --noise_type asymmetric --noise_rate 0.4 --dataset dirtymnist  --lr 1e-4

python f-correction.py --noise_type symmetric --noise_rate 0.8 --epoch 20 --dataset dirtymnist
python f-correction.py --noise_type asymmetric --noise_rate 0.4 --epoch 20 --dataset dirtymnist
python f-correction.py --noise_type symmetric --noise_rate 0.5 --epoch 20 --dataset dirtymnist
python f-correction.py --noise_type symmetric --noise_rate 0.2 --epoch 20 --dataset dirtymnist

python Co-teaching.py --noise_type symmetric --noise_rate 0.2 --model_type coteaching --dataset dirtymnist 
python Co-teaching.py --noise_type symmetric --noise_rate 0.5 --model_type coteaching --dataset dirtymnist 
python Co-teaching.py --noise_type symmetric --noise_rate 0.8 --model_type coteaching --dataset dirtymnist 
python Co-teaching.py --noise_type asymmetric --noise_rate 0.4  --model_type coteaching --dataset dirtymnist 

python Co-teaching.py --noise_type symmetric --noise_rate 0.2 --dataset dirtymnist 
python Co-teaching.py --noise_type symmetric --noise_rate 0.5 --dataset dirtymnist 
python Co-teaching.py --noise_type symmetric --noise_rate 0.8 --dataset dirtymnist 
python Co-teaching.py --noise_type asymmetric --noise_rate 0.4 --dataset dirtymnist 

python JoCoR.py --noise_type symmetric --noise_rate 0.2 --dataset dirtymnist
python JoCoR.py --noise_type symmetric --noise_rate 0.5 --dataset dirtymnist
python JoCoR.py --noise_type symmetric --noise_rate 0.8 --dataset dirtymnist
python JoCoR.py --noise_type asymmetric --noise_rate 0.4  --dataset dirtymnist


# python noise-adaptation.py --noise_type symmetric --noise_rate 0.8 
# python noise-adaptation.py --noise_type asymmetric --noise_rate 0.4 
# python noise-adaptation.py --noise_type symmetric --noise_rate 0.8 --dataset cifar10
# python noise-adaptation.py --noise_type asymmetric --noise_rate 0.4 --dataset cifar10

# python f-correction.py --noise_rate 0.4 --dataset mnist --epochs 20 --noise_type instance
# python f-correction.py --noise_rate 0.6 --dataset mnist --epochs 20 --noise_type instance
# python f-correction.py --noise_rate 0.4 --dataset cifar10 --epochs 200 --noise_type instance
# python f-correction.py --noise_rate 0.6 --dataset cifar10 --epochs 200 --noise_type instance

# python noise-adaptation.py --noise_type symmetric --noise_rate 0.8 --lr_noise 1e-4 --wd_noise 0.1 --beta 0.9
# python noise-adaptation.py --noise_type asymmetric --noise_rate 0.4

# python noise-adaptation.py --noise_type symmetric --noise_rate 0.8 --dataset cifar10 --lr_noise 1e-4 --wd_noise 0.1 --beta 0.9
# python noise-adaptation.py --noise_type asymmetric --noise_rate 0.4 --dataset cifar10


