python train.py \
  --task=retraining \
  --pruning-type=batchnorm \
  --uniform \
  --hyp=data/hyp.retraining.yaml \
  --epochs=20 \
  --save-weight \
  --batch-size=16 \
  --test-size=0.2 \
  --workers=8 \
