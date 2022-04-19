cd ..
python noise-adaptation.py --noise_type symmetric --noise_rate 0.2 --gpu 4 --dataset cifar100
python noise-adaptation.py --noise_type symmetric --noise_rate 0.5 --gpu 4 --dataset cifar100
python noise-adaptation.py --noise_type symmetric --noise_rate 0.8 --gpu 4 --dataset cifar100 
python noise-adaptation.py --noise_type asymmetric --noise_rate 0.4 --gpu 4 --dataset cifar100
