cd ..
python noise-adaptation.py --noise_type symmetric --noise_rate 0.2 --dataset mnist
python noise-adaptation.py --noise_type symmetric --noise_rate 0.5 --dataset mnist
python noise-adaptation.py --noise_type symmetric --noise_rate 0.8 --dataset mnist 
python noise-adaptation.py --noise_type asymmetric --noise_rate 0.4 --dataset mnist

python JoCoR.py --noise_type symmetric --noise_rate 0.2 --gpu 1
python JoCoR.py --noise_type symmetric --noise_rate 0.5 --gpu 1
python JoCoR.py --noise_type symmetric --noise_rate 0.8 --gpu 1
python JoCoR.py --noise_type asymmetric --noise_rate 0.4  --gpu 1
