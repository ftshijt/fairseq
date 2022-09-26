. ./tools/activate_python.sh

echo '''
python fairseq_cli/hydra_train.py \
  --config-dir examples/hubert/config/finetune \
  --config-name base_10h \
  task.data=/mnt/d/jiatong/data/LibriSpeech/formated/wavelist/sized_list task.label_dir=/mnt/d/jiatong/data/LibriSpeech/formated/trans/char \

python fairseq_cli/hydra_train.py \
  --config-dir examples/hubert/config/finetune \
  --config-name base_10h_multires \
  task.data=/mnt/d/jiatong/data/LibriSpeech/formated/wavelist/sized_list task.label_dir=/mnt/d/jiatong/data/LibriSpeech/formated/trans/char \
  model.w2v_path=/mnt/d/jiatong/data/label_rate50/checkpoint_last.pt

python fairseq_cli/hydra_train.py \
  --config-dir examples/hubert/config/finetune \
  --config-name base_10h_multires \
  task.data=/mnt/d/jiatong/data/LibriSpeech/formated/wavelist/sized_list task.label_dir=/mnt/d/jiatong/data/LibriSpeech/formated/trans/char \
  model.w2v_path="/mnt/d/jiatong/data/label_rate25/checkpoint_last.pt--/mnt/d/jiatong/data/label_rate33/checkpoint_last.pt" \
  model.label_rates=\'4,3\'

echo '''
python fairseq_cli/hydra_train.py \
  --config-dir examples/hubert/config/finetune \
  --config-name base_10h_multires \
  task.data=/mnt/d/jiatong/data/LibriSpeech/formated/wavelist/sized_list task.label_dir=/mnt/d/jiatong/data/LibriSpeech/formated/trans/char \
  model.w2v_path="/mnt/d/jiatong/data/label_rate10/checkpoint_last.pt--/mnt/d/jiatong/data/label_rate25/checkpoint_last.pt--/mnt/d/jiatong/data/label_rate33/checkpoint_last.pt--/mnt/d/jiatong/data/label_rate50/checkpoint_last.pt" \
  model.label_rates=\'5,2,2,3,1,2\'


python fairseq_cli/hydra_train.py \
  --config-dir examples/hubert/config/finetune \
  --config-name base_10h_multires \
  task.data=/mnt/d/jiatong/data/LibriSpeech/formated/wavelist/sized_list task.label_dir=/mnt/d/jiatong/data/LibriSpeech/formated/trans/char \
  model.w2v_path="/mnt/d/jiatong/data/label_rate33/checkpoint_last.pt--/mnt/d/jiatong/data/label_rate50/checkpoint_last.pt" \
  model.label_rates=\'3,2\'
