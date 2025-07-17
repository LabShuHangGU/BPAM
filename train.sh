export MKL_NUM_THREADS=8
export NUMEXPR_NUM_THREADS=8
export OMP_NUM_THREADS=8
CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/train_tone.yml
# CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/train_tone_4k.yml
# CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/train_fivek_pr.yml
# CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/train_ppr10k.yml
# CUDA_VISIBLE_DEVICES=0 python train.py -opt options/train/train_lcdp.yml
