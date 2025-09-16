---


python main.py --config .\configs\MIMO.yaml --model_name WMMSE --T 128 --I 16 --d 1 --snr 0
python main.py --config .\configs\MIMO.yaml --model_name MMMSE --T 128 --I 16 --d 1 --snr 0
python main.py --config .\configs\MIMO.yaml --model_name R_WMMSE --T 128 --I 16 --d 1 --snr 0
python main.py --config .\configs\MIMO.yaml --model_name R_MMMSE --T 128 --I 16 --d 1 --snr 0
python main.py --config .\configs\MIMO.yaml --model_name A_MMMSE --T 128 --I 16 --d 1 --snr 0 --omega 0.6 --lr 0.4
python main.py --config .\configs\MIMO.yaml --model_name Nonhomo_QT --T 128 --I 16 --d 1 --snr 0


python main.py --config .\configs\MIMO.yaml --model_name WMMSE --T 256 --I 20 --d 4 --snr 10
python main.py --config .\configs\MIMO.yaml --model_name MMMSE --T 256 --I 20 --d 4 --snr 10
python main.py --config .\configs\MIMO.yaml --model_name R_WMMSE --T 256 --I 20 --d 4 --snr 10
python main.py --config .\configs\MIMO.yaml --model_name R_MMMSE --T 256 --I 20 --d 4 --snr 10
python main.py --config .\configs\MIMO.yaml --model_name A_MMMSE --T 256 --I 20 --d 4 --snr 10 --omega 0.8 --lr 0.05




