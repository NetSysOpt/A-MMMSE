---


python main_gpu.py --config .\configs\MIMO.yaml --model_name R_WMMSE_GPU --T 256 --I 20 --d 4 --snr 10
python main_gpu.py --config .\configs\MIMO.yaml --model_name A_MMMSE_GPU --T 256 --I 20 --d 4 --snr 10 --omega 0.4 --lr 0.1


python main_gpu.py --config .\configs\MIMO.yaml --model_name R_WMMSE_GPU --T 512 --I 20 --d 4 --snr 10
python main_gpu.py --config .\configs\MIMO.yaml --model_name A_MMMSE_GPU --T 512 --I 20 --d 4 --snr 10 --omega 0.4 --lr 0.1

python main_gpu.py --config .\configs\MIMO.yaml --model_name R_WMMSE_GPU --T 1024 --I 20 --d 4 --snr 10
python main_gpu.py --config .\configs\MIMO.yaml --model_name A_MMMSE_GPU --T 1024 --I 20 --d 4 --snr 10 --omega 0.4 --lr 0.1

python main_gpu.py --config .\configs\MIMO.yaml --model_name R_WMMSE_GPU --T 2048 --I 20 --d 4 --snr 10
python main_gpu.py --config .\configs\MIMO.yaml --model_name A_MMMSE_GPU --T 2048 --I 20 --d 4 --snr 10 --omega 0.4 --lr 0.1

