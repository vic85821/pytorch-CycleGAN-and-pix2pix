set -ex
python train.py --dataroot ../makeup_dataset/images/ --maskroot ../makeup_dataset/segs/ --name beautyGAN --model cycle_gan