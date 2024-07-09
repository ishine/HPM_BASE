'''
python -m torch.distributed.launch --nproc_per_node=2 --master_port=29501 /workspace/HPM_BASE/model/train_FastSpeech2.py

'''


import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2,3"
os.environ['OMP_NUM_THREADS'] = '4'
import sys
sys.path.append("/workspace/HPM_BASE/utils")
sys.path.append("/workspace/HPM_BASE/model")
sys.path.append("/workspace/HPM_BASE/transformer")
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import matplotlib
matplotlib.use('Agg')
import json
import numpy as np
from attrdict import AttrDict
import wandb
import yaml
from tqdm import tqdm


from torch.utils.checkpoint import checkpoint
#torch
import torch
import torch.nn.functional as F
import torch.nn as nn
from apex.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist
from apex import amp
from utils.hifigan_16_models import Generator
from transformer import Decoder,PostNet
from Dataset import Dataset
from torch.utils.data import DataLoader
from utils.tools import tool_to_device , synth_one_sample,log
from optimizer import ScheduledOptim

from utils.stft import TorchSTFT
from utils.env import AttrDict
from utils.hifigan_16_models import Generator


from FastSpeech2 import FastSpeech2, FastSpeech2Loss

with open('/workspace/HPM_BASE/config/fast_speech/preprocess.yaml') as f:
    preprocess_config = yaml.load(f, Loader=yaml.FullLoader)

with open('/workspace/HPM_BASE/config/fast_speech/model.yaml') as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)
with open('/workspace/HPM_BASE/config/fast_speech/train.yaml') as f:
    train_config = yaml.load(f, Loader=yaml.FullLoader)

print("config load")
wandb_flag = True
def load_checkpoint(model, optimizer,amp, checkpoint_path):
    state = torch.load(checkpoint_path)
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    amp.load_state_dict(state['amp'])
    return state['step']
def save_checkpoint(model, optimizer,amp, step, config):
    state = {
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'amp': amp.state_dict(),
        'step': step,
        'config': config
    }
    if wandb_flag:
        checkpoint_path = f"checkpoint_fast_speech_{step}.pth"
        torch.save(state, checkpoint_path)
        artifact = wandb.Artifact(f"checkpoint_{step}", type="model")
        artifact.add_file(checkpoint_path)
        wandb.log_artifact(artifact)
        wandb.save('train_FastSpeech2.py')
        wandb.config.update({'model_structure':str(model)})
    print(f"Saved checkpoint at step {step}")
def get_vocoder():
        checkpoint_path= "/workspace/HPM_BASE/vocoder/HiFi_GAN_16"
        name = "HiFi_GAN_16"
        config_file = os.path.join(checkpoint_path, "config.json")
        with open(config_file) as f:
            data = f.read()
        json_config = json.loads(data)
        h = AttrDict(json_config)
        torch.manual_seed(h.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(h.seed)

        vocoder = Generator(h).cuda()
        state_dict_g = torch.load(os.path.join(checkpoint_path, "g_HPM_Chem"), map_location="cuda")
        vocoder.load_state_dict(state_dict_g['generator'])
        vocoder.eval()
        vocoder.remove_weight_norm()
        return vocoder
def vocoder_infer(mels, vocoder, model_config, preprocess_config, lengths=None):
    name = model_config["vocoder"]["model"]  # HiFi_GAN_16
    with torch.no_grad():
        if name == "MelGAN":
            wavs = vocoder.inverse(mels / np.log(10))
        elif name == "HiFi-GAN":
            wavs = vocoder(mels).squeeze(1)  # torch.Size([1, 80, 448])
        elif name == "HiFi_GAN_16":
            wavs = vocoder(mels).squeeze(1)
        elif name == "HiFi_GAN_220":
            wavs = vocoder(mels).squeeze(1)
        elif name == "ISTFTNET":
            stft = TorchSTFT(filter_length=16, hop_length=4, win_length=16).cuda()
            spec, phase = vocoder(mels)
            y_g_hat = stft.inverse(spec, phase)
            wavs = y_g_hat.squeeze(1)

    wavs = (
            wavs.cpu().numpy()
            * preprocess_config["preprocessing"]["audio"]["max_wav_value"]
    ).astype("int16")
    wavs = [wav for wav in wavs]

    for i in range(len(mels)):
        if lengths is not None:
            wavs[i] = wavs[i][: lengths[i]]

    return wavs

# GPU 0번에 대한 상태 로깅 함수
def log_gpu_status():
    gpu_status = {}
    gpu_id = 0
    gpu_status[f"GPU_{gpu_id}_Memory_Used"] = torch.cuda.memory_allocated(gpu_id)
    gpu_status[f"GPU_{gpu_id}_Memory_Total"] = torch.cuda.get_device_properties(gpu_id).total_memory
    if wandb_flag:
        wandb.log(gpu_status)
def main():
    print("main")
    local_rank = int(os.environ['LOCAL_RANK'])
    print(f'local_rank: {local_rank}')
    if local_rank==0:
        if wandb_flag:
            wandb.init(project="fastSpeech2", config={**train_config,**model_config})
            # GPU 상태 설정
            wandb.config.log_gpu_status = True

    torch.cuda.set_device(local_rank)  
    dist.init_process_group(backend='nccl', init_method='env://')
          
    device = torch.device(f'cuda:{local_rank}')
    print(f'Running on device: {device}')
    print("main function")
    train_dataset = Dataset(
        "/workspace/DATA/chem/preprocessed_3_set2/train.txt",
        preprocess_config,
        train_config["optimizer"]["batch_size"],
    )
    train_sampler = DistributedSampler(train_dataset)
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4
    assert batch_size * group_size < len(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        sampler=train_sampler,
        batch_size=batch_size * group_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,   
        collate_fn=train_dataset.collate_fn,
    )

    model = FastSpeech2(preprocess_config, model_config).to(device)
    if list(model.parameters()):
        optimizer = ScheduledOptim(model, train_config, model_config, current_step=0)
        model, optimizer = amp.initialize(model, optimizer, opt_level="O2")
        model = DDP(model)
    
    Loss = FastSpeech2Loss(preprocess_config, model_config).to(device)
    vocoder = get_vocoder().to(device)


    train_log_path = "/workspace/HPM_BASE/log/FastSpeech2/train"
    os.makedirs(train_log_path, exist_ok=True)

    step = 0
    epoch =0

    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    synth_step = train_config["step"]["synth_step"]
    
    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n =0
    outer_bar.update()
    if wandb_flag and local_rank==0:
        wandb.watch(model)

    while True:
        epoch += 1
        torch.cuda.empty_cache()
        train_sampler.set_epoch(epoch) 
        for batchs in train_loader:
            for batch in batchs:
                batch = tool_to_device(batch, device)

                # forward
                output = model(*batch)
                # cal loss
                loss = Loss(batch, output)
                total_loss = loss[0]

                #Backward
                total_loss = total_loss / grad_acc_step

                with amp.scale_loss(total_loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                if step % grad_acc_step == 0:
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()

                if wandb_flag and local_rank==0 and step % log_step==0:
                    current_lr = optimizer.get_lr()
                    log_dict = {
                    "learning_rate": current_lr,
                    "step": step,
                    "epoch": epoch,
                    "loss": total_loss.item(),
                    "mel_loss": loss[1].item(),
                    "postnet_mel_loss": loss[2].item(),
                    "pitch_loss": loss[3].item(),
                    "energy_loss": loss[4].item(),
                    "duration_loss": loss[5].item(),                    
                    }
                # wandb logging
                    wandb.log(log_dict)
                if step % synth_step == 0:
                    if local_rank==0:
                        log_gpu_status()
                    torch.cuda.empty_cache()
                    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
                    

                    if wandb_flag and local_rank==0:
                        fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                            batch,
                            output,
                            vocoder,
                            model_config,
                            preprocess_config,
                        )
                        fig.savefig('/workspace/fast_speech/model/synth_sample.png',dpi=300)
                        wandb.log({
                            "Synth_Sample": [
                                wandb.Image('/workspace/fast_speech/model/synth_sample.png',caption=f"Step {step} Synthesized")
                            ],
                            "Reconstructed_Audio": wandb.Audio(wav_reconstruction, sample_rate=sampling_rate, caption=f"Step {step} Reconstructed"),
                            "Synthesized_Audio": wandb.Audio(wav_prediction, sample_rate=sampling_rate, caption=f"Step {step} Synthesized")
                        })
                    
                if step% 150000 ==0 and step!=0:
                    if local_rank ==0:
                        combined_config = {**train_config, **model_config}
                        save_checkpoint(model, optimizer,amp, step, combined_config)
                if step == total_step:
                    if wandb_flag and local_rank==0:
                        wandb.finish()
                    quit()

                step += 1
                outer_bar.update(1)               
if __name__ == "__main__":
    print("main")

    main()
    