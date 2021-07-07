# gpt2-bengali
Currently, there is no GPT2 model that was trained from scratch for Bengali on the hub. For this project, the goal is to create a strong language generation model for Bengali using GPT2 Model.

# Pretraining Process
```
huggingface-cli repo create norwegian-gpt2
```

Next we clone the model repository to add the tokenizer and model files.

```
git clone https://huggingface.co/<your-username>/gpt2-bengali
```

To ensure that all tensorboard traces will be uploaded correctly, we need to 
track them. You can run the following command inside your model repo to do so.

```
cd gpt2-bengali
git lfs track "*tfevents*"
```

Great, we have set up our model repository. During training, we will automatically
push the training logs and model weights to the repo.

Next, let's add a symbolic link to the `run_clm_flax.py`.

```bash
export MODEL_DIR="./gpt2-bengali"
ln -s ~/transformers/examples/flax/language-modeling/run_clm_flax.py run_clm_flax.py
```

Next, we'll follow the same steps as above in [Train tokenizer](#train-tokenizer) to train the tokenizer.

### Create configuration

Next, we create the model's configuration file. This is as simple 
as loading and storing [`**gpt2**`](https://huggingface.co/gpt2)
in the local model folder:

```python
from transformers import GPT2Config

model_dir = "./gpt2-bengali"  # ${MODEL_DIR}

config = GPT2Config.from_pretrained("gpt2", resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0)
config.save_pretrained(model_dir)
```

### Train model

Next we can run the example script to pretrain the model:

```bash
./run_clm_flax.py \
    --output_dir="${MODEL_DIR}" \
    --model_type="gpt2" \
    --config_name="${MODEL_DIR}" \
    --tokenizer_name="${MODEL_DIR}" \
    --dataset_name="mc4" \
    --dataset_config_name="bn" \
    --do_train --do_eval \
    --block_size="512" \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="64" \
    --learning_rate="5e-3" --warmup_steps="1000" \
    --adam_beta1="0.9" --adam_beta2="0.98" --weight_decay="0.01" \
    --overwrite_output_dir \
    --num_train_epochs="20" \
    --logging_steps="500" \
    --save_steps="2500" \
    --eval_steps="2500" \
    --push_to_hub
```
