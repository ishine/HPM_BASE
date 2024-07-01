# Input
# Phoneme Feature
# Speaker Feature
# Lip-M Feature
# 만 넣어서 Mel Generator -> Vocoder 수행
import distutils
import distutils.version
distutils.version.LooseVersion = lambda x: x
import wandb
import os
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
from scipy.io import wavfile
from matplotlib import pyplot as plt
import sys
# os.environ["CUDA_VISIBLE_DEVICES"] = "7"

sys.path.append("/workspace/HPM_BASE/model")
sys.path.append("/workspace/HPM_BASE/transformer")
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.hifigan_16_models import Generator
from attrdict import AttrDict
import wandb
# from modules_1 import Multi_head_Duration_Aligner
from Multi_head_Duration_Aligner import Multi_head_Duration_Aligner
from transformer import Decoder,PostNet
device = torch.device("cuda")
if torch.cuda.device_count() > 1:
    print(f"Using {torch.cuda.device_count()} GPUs")
# print(f"CUDA Available: {torch.cuda.is_available()}")
# print(f"Device Count: {torch.cuda.device_count()}")
# wandb.init(project ="HPM_Dubb")

from Dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import yaml
from utils.tools import tool_to_device , synth_one_sample,log
from tqdm import tqdm
from optimizer import ScheduledOptim
from torch.utils.tensorboard import SummaryWriter
from utils.stft import TorchSTFT
from utils.env import AttrDict
from utils.hifigan_16_models import Generator
from utils.istft_models import istft_Generator
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
# from tensorboardX import SummaryWriter
with open('/workspace/HPM_BASE/config/Chem/preprocess.yaml') as f:
    preprocess_config = yaml.load(f, Loader=yaml.FullLoader)

with open('/workspace/HPM_BASE/config/Chem/model.yaml') as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)
#prepare model 
# model, optimizer = get_model()#(configs, device, train=True)

wandb_flag = True
# def get_model():
def get_mask_from_lengths(lengths, max_len=None):
    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = torch.max(lengths).item()

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = ids >= lengths.unsqueeze(1).expand(-1, max_len)

    return mask
class CTC_classifier_mel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.classifier = nn.Linear(256, num_classes)
    def forward(self, Dub):
        size = Dub.size()
        Dub = Dub.reshape(-1, size[2]).contiguous()
        Dub = self.classifier(Dub)
        return Dub.reshape(size[0], size[1], -1) 
class HPM_Dubb_1(nn.Module):
    def __init__(self,model_config,preprocess_config):
        super(HPM_Dubb_1, self).__init__()
        
        # Multi head duration aligner 
        self.MDA = Multi_head_Duration_Aligner(model_config)
        # self.proj_fusion = nn.Conv1d(768, 256, kernel_size=1, padding=0, bias=False)
        self.postnet = PostNet()
        self.proj_fusion = nn.Conv1d(256, 256, kernel_size=1, padding=0, bias=False)
        self.decoder = Decoder(model_config)
        self.mel_linear = nn.Linear(
            model_config["transformer"]["decoder_hidden"],
            preprocess_config["preprocessing"]["mel"]["n_mel_channels"],
        )
        self.CTC_classifier_mel = CTC_classifier_mel(model_config["Symbols"]["phonemenumber"])  # len(symbols)

        self.Synchronization_coefficient = 4  
        """
        ===============================
        Q&A: Why is the Synchronization_coefficient set 4, can I change it using another positive integer? 
             Follow the Formula: n = \frac{T_{mel}}{T_v}=\frac{sr/hs}{FPS} \in \mathbb{N}^{+}.
             e.g., in our paper, for chem dataset, we set the sr == 16000Hz, hs == 160, win == 640, FPS == 25, so n is 4.
                                for chem dataset, we set the sr == 22050Hz, hs == 220, win == 880, FPS == 25, so n is 4.009. (This is the meaning of the approximately equal sign in the article). 
        """
    """
    forward에 대한 입력은 데이터 로더에서 배치단위로 제공되어야 한다. 
    collate_fn 에서 반환하는 값들이 입력인자와 일치해야한다.
    """
    def forward(
            # self,
            # speaker,
            # texts,
            # src_lens,
            # max_src_len,
            # mel_lens=None,
            # max_mel_len=None,
            # lip_embedding=None,
            # lip_lens=None,
            # max_lip_lens=None,
            # spks=None,
            self,
            ids,#0
            raw_texts,#1
            speakers,#2
            texts,#3
            src_lens,#4
            max_src_len,#5
            mels,#6
            mel_lens,#7
            max_mel_len,#8
            pitches,
            energies,
            durations,
            # spks,
            # emotions,
            # emos,
            # feature_256,
            lip_lens,
            max_lip_lens,
            lip_embedding,
            ):
        useGT=True
        src_masks = get_mask_from_lengths(src_lens, max_src_len) 
        lip_masks = get_mask_from_lengths(lip_lens, max_lip_lens)
        if useGT:
            # print(f"HPM_Dubb_1 : max_mel_len : {max_mel_len}")
            mel_masks = (
                get_mask_from_lengths(mel_lens, max_mel_len)
            )
        else:
            # print(f"HPM_Dubb_1 : max_mel_len : {max_mel_len}")
            mel_masks = (
                get_mask_from_lengths(lip_lens*self.Synchronization_coefficient, max_lip_lens*self.Synchronization_coefficient)
            )
        #lip_embedding,lip_masks,texts,src_masks,max_src_len,lip_lens,src_lens,reference_embedding = None,
        
        output, ctc_pred_MDA_video = self.MDA(lip_embedding,lip_masks,texts,src_masks,max_src_len,lip_lens,src_lens)#,reference_embedding= spks)
        # 길이 조정(추가사항)
                # 길이 조정
        B, L, D = output.size()
        target_length = max_mel_len
        output = output.transpose(1, 2)  # [B, D, L]
        output = F.interpolate(output, size=target_length, mode='nearest')
        output = output.transpose(1, 2)  # [B, target_length, D]
        # print("[MDA]interpolated output shape:", output.shape)
        # ctc_pred_MDA_video도 같은 방식으로 조정
        ctc_pred_MDA_video = ctc_pred_MDA_video.transpose(1, 2)
        ctc_pred_MDA_video = F.interpolate(ctc_pred_MDA_video, size=target_length, mode='nearest')
        ctc_pred_MDA_video = ctc_pred_MDA_video.transpose(1, 2)
        # output = F.interpolate(output.transpose(1, 2), scale_factor=self.Synchronization_coefficient, mode='nearest').transpose(1, 2)
        # Following the paper concatenation the three information
        fusion_output = torch.cat([output], dim=-1)
        fusion_output = self.proj_fusion(fusion_output.transpose(1, 2)).transpose(1, 2)


        """=========Mel-Generator========="""
        # print("[proj_fusion]fusion_output shape:", fusion_output.shape)
        # print("mel_masks shape:", mel_masks.shape)

        fusion_output, mel_masks = self.decoder(fusion_output, mel_masks)
        # print("[decoder]fusion_output shape:", fusion_output.shape)
        ctc_pred_mel = self.CTC_classifier_mel(fusion_output)
        # print("[ctc_classifier_mel]ctc_pred_mel shape:", ctc_pred_mel.shape)
        fusion_output = self.mel_linear(fusion_output)
        # print("[mel_linear]fusion_output shape:", fusion_output.shape)
        postnet_output = self.postnet(fusion_output) + fusion_output
        
        ctc_loss_all = [ctc_pred_MDA_video, ctc_pred_mel]
        
        
        #Using Output

        #Mel Decoder 
        #Mel Linear

        return (
            fusion_output,
            postnet_output,
            src_masks, 
            mel_masks, 
            src_lens,
            lip_lens*self.Synchronization_coefficient,
            ctc_loss_all)


class HPM_Dubb_1_Loss(nn.Module):
    def __init__(self, preprocess_config, model_config):
        super(HPM_Dubb_1_Loss, self).__init__()
        self.mae_loss = nn.L1Loss()
        self.mse_loss = nn.MSELoss()
        self.CTC_criterion = nn.CTCLoss(blank=0, zero_infinity=True, reduction='sum').cuda()

    def weights_nonzero_speech(self, target):
        dim = target.size(-1)
        return target.abs().sum(-1, keepdim=True).ne(0).float().repeat(1, 1, dim)

    def mse_loss_v2c(self, decoder_output, target):
        assert decoder_output.shape == target.shape
        mse_loss = F.mse_loss(decoder_output, target, reduction='none')
        weights = self.weights_nonzero_speech(target)
        mse_loss = (mse_loss * weights).sum() / weights.sum()
        return mse_loss

    def forward(self, inputs, predictions):
        (
            _,
            _,
            _,
            texts,
            src_lens,
            max_src_len,
            mel_targets,
            mel_lens,
            max_mel_len,
            pitch_targets,#prosody adaptor에서
            energy_targets,
            duration_targets,
            # _,#spk
            lip_lens,
            max_lip_lens,
            _,
        ) = inputs

        (
            mel_predictions,
            postnet_mel_predictions,
            src_masks,
            mel_masks,
            _,
            _,
            ctc_loss_all,
        ) = predictions
        '''
                    fusion_output,
                    postnet_output,
                    src_masks, 
                    mel_masks, 
                    src_lens,
                    lip_lens*self.Synchronization_coefficient,
                    ctc_loss_all
        '''
        src_masks = ~src_masks
        mel_masks = ~mel_masks
        # Clamp mel_lens to a maximum of 1000
        max_length = 1000
        # mel_lens = torch.clamp(mel_lens, max=max_length)
        mel_targets = mel_targets[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]
        mel_targets.requires_grad = False

        ctc_pred_MDA_video = ctc_loss_all[0]
        ctc_pred_mel = ctc_loss_all[1]
        # print("[loss input]ctc_pred_mel shape:", ctc_pred_mel.shape)
        CTC_loss_MDA_video = self.CTC_criterion(ctc_pred_MDA_video.transpose(0, 1).log_softmax(2), texts, lip_lens, src_lens) / texts.shape[0]
        CTC_loss_MEL = self.CTC_criterion(ctc_pred_mel.transpose(0, 1).log_softmax(2), texts, mel_lens, src_lens) / texts.shape[0]

        mse_loss_v2c1 = self.mse_loss_v2c(mel_predictions, mel_targets)
        mse_loss_v2c2 = self.mse_loss_v2c(postnet_mel_predictions, mel_targets)       
        
        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        postnet_mel_predictions = postnet_mel_predictions.masked_select(mel_masks.unsqueeze(-1))

        mel_targets = mel_targets.masked_select(mel_masks.unsqueeze(-1))
        mel_loss = self.mae_loss(mel_predictions, mel_targets)
        postnet_mel_loss = self.mae_loss(postnet_mel_predictions, mel_targets)
        
        #pitch, energy
        # total_loss = L_s
        # L_s = lambda_1*L_mel + L_postnet + L_v2c1 + L_v2c2 + L_ctc1 + L_ctc2
        total_loss = (
            1.3*mel_loss + 
            1.3*postnet_mel_loss + 
            mse_loss_v2c1 + 
            mse_loss_v2c2 + 
            0.01 * CTC_loss_MDA_video 
            + 0.01 * CTC_loss_MEL
        )

        return (
            total_loss,
            mel_loss,
            postnet_mel_loss,
            mse_loss_v2c1,
            mse_loss_v2c2,
            0.01 * CTC_loss_MDA_video,
            0.01 * CTC_loss_MEL,
        )

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

        vocoder = Generator(h).to(device)
        state_dict_g = torch.load(os.path.join(checkpoint_path, "g_HPM_Chem"), map_location=device)
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
            stft = TorchSTFT(filter_length=16, hop_length=4, win_length=16).to(device)
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

# GPU 상태 로깅 함수
# GPU 0번에 대한 상태 로깅 함수
def log_gpu_status():
    gpu_status = {}
    gpu_id = 0
    gpu_status[f"GPU_{gpu_id}_Memory_Used"] = torch.cuda.memory_allocated(gpu_id)
    gpu_status[f"GPU_{gpu_id}_Memory_Total"] = torch.cuda.get_device_properties(gpu_id).total_memory
    if wandb_flag:
        wandb.log(gpu_status)
def main():

# wandb 초기화
    train_config = yaml.load(
        open("/workspace/HPM_BASE/config/Chem/train.yaml", "r"), Loader=yaml.FullLoader
    )
    if wandb_flag:
        wandb.init(project="HPM_step1_test1", config=train_config)
        # GPU 상태 설정
        wandb.config.log_gpu_status = True

    flag = True
    print("main function")
    # wandb.init(project='HPM_TEST_1')
    #/workspace/DATA/chem/processed

    train_dataset = Dataset(
        "/workspace/DATA/chem/preprocessed_3/train.txt",
        preprocess_config,
        train_config["optimizer"]["batch_size"],
    )
    # val_dataset = Dataset(
    #     "/workspace/DATA/chem/processed/val.txt", 
    #     preprocess_config, 
    #     train_config["optimizer"]["batch_size"],
    # )
    batch_size = train_config["optimizer"]["batch_size"]
    group_size = 4
    assert batch_size * group_size < len(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size * group_size,
        shuffle=True,
        num_workers=1,
        pin_memory=True,   
        collate_fn=train_dataset.collate_fn,
    )
    # val_loader = DataLoader(
    #     val_dataset,
    #     batch_size=train_config["optimizer"]["batch_size"],
    #     shuffle=False,
    #     collate_fn=val_dataset.collate_fn,
    # )

    #model and optimizer
    model = HPM_Dubb_1(model_config,preprocess_config)
    optimizer = ScheduledOptim(model, train_config, model_config,current_step=0)
    model = nn.DataParallel(model)
    # num_param = sum(param.numel() for param in model.parameters())
    # print("Number of HPM_Dubbing Parameters:", num_param)
    Loss = HPM_Dubb_1_Loss(preprocess_config, model_config).to(device)
    vocoder = get_vocoder()
    # logger 
    train_log_path = "/workspace/HPM_BASE/model/log/Chem/train"
    os.makedirs(train_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path, flush_secs=10)
    # Training
    step =0
    epoch = 0
    grad_acc_step = train_config["optimizer"]["grad_acc_step"]
    grad_clip_thresh = train_config["optimizer"]["grad_clip_thresh"]
    total_step = train_config["step"]["total_step"]
    log_step = train_config["step"]["log_step"]
    save_step = train_config["step"]["save_step"]
    synth_step = train_config["step"]["synth_step"]
    val_step = train_config["step"]["val_step"]
    loss_model = model_config["loss_function"]["model"]

    outer_bar = tqdm(total=total_step, desc="Training", position=0)
    outer_bar.n =0
    outer_bar.update()
    if wandb_flag:
        wandb.watch(model)
    while True:
        epoch += 1
        for batchs in train_loader:
            for batch in batchs:
                batch = tool_to_device(batch, device)
                
                #Forward
                output = model(*batch)
                #Cal Loss 
                losses = Loss(batch, output)
                total_loss = losses[0]
                """
                        return (
                    total_loss,
                    mel_loss,
                    mse_loss_v2c,
                    0.01 * CTC_loss_MDA_video,
                    0.01 * CTC_loss_MEL,
                )
                """

                #Backward
                total_loss = total_loss / grad_acc_step
                total_loss.backward()

                if wandb_flag and step % log_step==0:
                # wandb logging
                    wandb.log({"loss": total_loss.item(), 
                            "mel_loss": losses[1].item(), 
                            "mse_loss_v2c": losses[2].item(), 
                            "CTC_loss_MDA_video": losses[3].item(), 
                            "CTC_loss_MEL": losses[4].item(),
                            "step": step,
                            "epoch": epoch})
                if step % grad_acc_step == 0:
                    # Clipping gradients to avoid gradient explosion
                    nn.utils.clip_grad_norm_(model.parameters(), grad_clip_thresh)

                    # Update weights
                    optimizer.step_and_update_lr()
                    optimizer.zero_grad()
                if step % synth_step == 0:
                    log_gpu_status()
                    fig, wav_reconstruction, wav_prediction, tag = synth_one_sample(
                        batch,
                        output,
                        vocoder,
                        model_config,
                        preprocess_config,
                    )
                    fig.savefig('synth_sample.png',dpi=300)

                    wandb.log({
                        "Synth_Sample": [
                            wandb.Image('synth_sample.png')
                        ]
                    })
                    '''
                    plt.imshow(attention_map, aspect='auto', origin='lower', interpolation='none')
                    plt.savefig('attention_map.png', figsize=(16, 4))

                    wandb.log({
                        "Attention Map": [
                            wandb.Image('attention_map.png')
                        ]
                    })
                    '''
                    sampling_rate = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
                    
                    # Log synthesized audio samples
                    wandb.log({
                        "Reconstructed_Audio": wandb.Audio(wav_reconstruction, sample_rate=sampling_rate, caption=f"Step {step} Reconstructed"),
                        "Synthesized_Audio": wandb.Audio(wav_prediction, sample_rate=sampling_rate, caption=f"Step {step} Synthesized")
                    })
                    # log(
                    #     train_logger,
                    #     fig=fig,
                    #     tag="Training/step_{}_{}".format(step, tag),
                    # )
                    # sampling_rate = preprocess_config["preprocessing"]["audio"][
                    #     "sampling_rate"
                    # ]

                    # log(
                    #     train_logger,
                    #     audio=wav_reconstruction,
                    #     sampling_rate=sampling_rate,
                    #     tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    # )
                    # log(
                    #     train_logger,
                    #     audio=wav_prediction,
                    #     sampling_rate=sampling_rate,
                    #     tag="Training/step_{}_{}_synthesized".format(step, tag),
                    # )
                if step == total_step:
                    if wandb_flag:
                        wandb.finish()
                    quit()

                step += 1
                outer_bar.update(1)

     

if __name__ == "__main__":
    print("main")

    main()
    