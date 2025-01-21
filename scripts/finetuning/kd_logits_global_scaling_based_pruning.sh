python train.py \
  --task=retraining \
  --pruning-type=batchnorm \
  --kd \
  --kd-type=logits \
  --hyp=data/hyp.knowledge_distillation.yaml \
  --epochs=20 \
  --save-weight \
  --batch-size=16 \
  --test-size=0.2 \
  --workers=8 \
