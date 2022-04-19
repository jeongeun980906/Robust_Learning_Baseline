cd ..
python noise-adaptation.py --noise_type symmetric --noise_rate 0.2 --dataset cifar10
python noise-adaptation.py --noise_type symmetric --noise_rate 0.5 --dataset cifar10
python noise-adaptation.py --noise_type symmetric --noise_rate 0.8 --dataset cifar10 
python noise-adaptation.py --noise_type asymmetric --noise_rate 0.4 --dataset cifar10

python JoCoR.py --noise_type symmetric --noise_rate 0.2 --dataset cifar10
python JoCoR.py --noise_type symmetric --noise_rate 0.5 --dataset cifar10
python JoCoR.py --noise_type symmetric --noise_rate 0.8 --dataset cifar10
python JoCoR.py --noise_type asymmetric --noise_rate 0.4  --dataset cifar10



