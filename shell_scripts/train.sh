. ./tools/activate_python.sh

python fairseq_cli/hydra_train.py \
  --config-dir examples/hubert/config/finetune \
  --config-name base_10h \
  task.data=/mnt/d/jiatong/data/LibriSpeech/formated/wavelist/sized_list task.label_dir=/mnt/d/jiatong/data/LibriSpeech/formated/trans/char \
  model.w2v_path=/mnt/d/jiatong/data/label_rate50/checkpoint_last.pt
