cd ..
python noise-adaptation.py --noise_type symmetric --noise_rate 0.2 --dataset dirtycifar10
python noise-adaptation.py --noise_type symmetric --noise_rate 0.5 --dataset dirtycifar10
python noise-adaptation.py --noise_type symmetric --noise_rate 0.8 --dataset dirtycifar10
python noise-adaptation.py --noise_type asymmetric --noise_rate 0.4 --dataset dirtycifar10

python f-correction.py --noise_type symmetric --noise_rate 0.8 --dataset dirtycifar10 --epochs 200
python f-correction.py --noise_type symmetric --noise_rate 0.2 --dataset dirtycifar10 --epochs 200
python f-correction.py --noise_type symmetric --noise_rate 0.5 --dataset dirtycifar10  --epochs 200
python f-correction.py --noise_type asymmetric --noise_rate 0.4 --dataset dirtycifar10  --epochs 200

python Co-teaching.py --noise_type symmetric --noise_rate 0.2 --model_type coteaching --dataset dirtycifar10 
python Co-teaching.py --noise_type symmetric --noise_rate 0.5 --model_type coteaching --dataset dirtycifar10 
python Co-teaching.py --noise_type symmetric --noise_rate 0.8 --model_type coteaching --dataset dirtycifar10 
python Co-teaching.py --noise_type asymmetric --noise_rate 0.4  --model_type coteaching --dataset dirtycifar10 

python Co-teaching.py --noise_type symmetric --noise_rate 0.2 --dataset dirtycifar10 
python Co-teaching.py --noise_type symmetric --noise_rate 0.5 --dataset dirtycifar10 
python Co-teaching.py --noise_type symmetric --noise_rate 0.8 --dataset dirtycifar10 
python Co-teaching.py --noise_type asymmetric --noise_rate 0.4 --dataset dirtycifar10 

python JoCoR.py --noise_type symmetric --noise_rate 0.2 --dataset dirtycifar10
python JoCoR.py --noise_type symmetric --noise_rate 0.5 --dataset dirtycifar10
python JoCoR.py --noise_type symmetric --noise_rate 0.8 --dataset dirtycifar10
python JoCoR.py --noise_type asymmetric --noise_rate 0.4  --dataset dirtycifar10

python Dividemix.py --noise_type 