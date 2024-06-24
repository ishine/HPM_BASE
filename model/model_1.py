# Input
# Phoneme Feature
# Speaker Feature
# Lip-M Feature
# 만 넣어서 Mel Generator -> Vocoder 수행
import wandb
import os
import json

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
from scipy.io import wavfile
from matplotlib import pyplot as plt
import sys
os.environ["CUDA_VISIBLE_DEVICES"] = "7"

sys.path.append("/workspace/CODE/Jenny_Test/model")
sys.path.append("/workspace/CODE/Jenny_Test/transformer")
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.hifigan_16_models import Generator
from attrdict import AttrDict
import wandb
# from modules_1 import Multi_head_Duration_Aligner
from Multi_head_Duration_Aligner import Multi_head_Duration_Aligner
from transformer import Decoder
device = torch.device("cuda")

# wandb.init(project ="HPM_Dubb")

from Dataset import Dataset
from torch.utils.data import DataLoader
import torch.nn as nn
import yaml
from utils.tools import to_device , synth_one_sample,log
from tqdm import tqdm
from optimizer import ScheduledOptim
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
with open('/workspace/CODE/Jenny_Test/config/Chem/preprocess.yaml') as f:
    preprocess_config = yaml.load(f, Loader=yaml.FullLoader)

with open('/workspace/CODE/Jenny_Test/config/Chem/model.yaml') as f:
    model_config = yaml.load(f, Loader=yaml.FullLoader)
#prepare model 
# model, optimizer = get_model()#(configs, device, train=True)


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
        self.proj_fusion = nn.Conv1d(768, 256, kernel_size=1, padding=0, bias=False)
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
            ids,
            raw_texts,
            speakers,
            texts,
            src_lens,
            max_src_len,
            mels,
            mel_lens,
            max_mel_len,
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
            mel_masks = (
                get_mask_from_lengths(mel_lens, max_mel_len)
            )
        else:
            mel_masks = (
                get_mask_from_lengths(lip_lens*self.Synchronization_coefficient, max_lip_lens*self.Synchronization_coefficient)
            )
        #lip_embedding,lip_masks,texts,src_masks,max_src_len,lip_lens,src_lens,reference_embedding = None,
        
        output, ctc_pred_MDA_video = self.MDA(lip_embedding,lip_masks,texts,src_masks,max_src_len,lip_lens,src_lens)#,reference_embedding= spks)
        # Following the paper concatenation the three information
        fusion_output = torch.cat([output], dim=-1)
        fusion_output = self.proj_fusion(fusion_output.transpose(1, 2)).transpose(1, 2)


        """=========Mel-Generator========="""
        fusion_output, mel_masks = self.decoder(fusion_output, mel_masks)
        ctc_pred_mel = self.CTC_classifier_mel(fusion_output)
        
        fusion_output = self.mel_linear(fusion_output)
        # postnet_output = self.postnet(fusion_output) + fusion_output
        
        ctc_loss_all = [ctc_pred_MDA_video, ctc_pred_mel]
        
        
        #Using Output

        #Mel Decoder 
        #Mel Linear

        return fusion_output,src_masks, mel_masks, ctc_loss_all


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
            mels,
            mel_lens,
            max_mel_len,
            _,
            _,
            _,
            _,
            lip_lens,
            max_lip_lens,
            _,
        ) = inputs

        (
            mel_predictions,
            src_masks,
            mel_masks,
            ctc_loss_all,
        ) = predictions

        src_masks = ~src_masks
        mel_masks = ~mel_masks

        mels = mels[:, : mel_masks.shape[1], :]
        mel_masks = mel_masks[:, :mel_masks.shape[1]]
        mels.requires_grad = False

        ctc_pred_MDA_video = ctc_loss_all[0]
        ctc_pred_mel = ctc_loss_all[1]

        CTC_loss_MDA_video = self.CTC_criterion(ctc_pred_MDA_video.transpose(0, 1).log_softmax(2), texts, lip_lens, src_lens) / texts.shape[0]
        CTC_loss_MEL = self.CTC_criterion(ctc_pred_mel.transpose(0, 1).log_softmax(2), texts, mel_lens, src_lens) / texts.shape[0]

        mse_loss_v2c = self.mse_loss_v2c(mel_predictions, mels)

        mel_predictions = mel_predictions.masked_select(mel_masks.unsqueeze(-1))
        mels = mels.masked_select(mel_masks.unsqueeze(-1))

        mel_loss = self.mae_loss(mel_predictions, mels)

        total_loss = (
            mel_loss + mse_loss_v2c + 0.01 * CTC_loss_MDA_video + 0.01 * CTC_loss_MEL
        )

        return (
            total_loss,
            mel_loss,
            mse_loss_v2c,
            0.01 * CTC_loss_MDA_video,
            0.01 * CTC_loss_MEL,
        )

def get_vocoder():
        checkpoint_path= "/workspace/CODE/Jenny_Test/vocoder/HiFi_GAN_16"
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


# GPU 상태 로깅 함수
# GPU 7번에 대한 상태 로깅 함수
def log_gpu_status():
    gpu_status = {}
    gpu_id = 7
    gpu_status[f"GPU_{gpu_id}_Memory_Used"] = torch.cuda.memory_allocated(gpu_id)
    gpu_status[f"GPU_{gpu_id}_Memory_Total"] = torch.cuda.get_device_properties(gpu_id).total_memory
    wandb.log(gpu_status)
def main():

# wandb 초기화
    train_config = yaml.load(
        open("/workspace/CODE/Jenny_Test/config/Chem/train.yaml", "r"), Loader=yaml.FullLoader
    )
    wandb.init(project="HPM_step1_test1", config=train_config)
    # GPU 상태 설정
    wandb.config.log_gpu_status = True

    flag = True
    print("main function")
    # wandb.init(project='HPM_TEST_1')
    #/workspace/DATA/chem/processed

    train_dataset = Dataset(
        "/workspace/DATA/chem/processed_test/train.txt",
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
        num_workers=16,
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
    train_log_path = "/workspace/CODE/Jenny_Test/model/log/Chem/train"
    os.makedirs(train_log_path, exist_ok=True)
    train_logger = SummaryWriter(train_log_path)
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
    wandb.watch(model)
    while True:
        epoch += 1
        for batchs in train_loader:
            for batch in batchs:
                batch = to_device(batch, device)
                #Forward
                output = model(*(batch))
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

                if step % log_step==0:
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
                    log(
                        train_logger,
                        fig=fig,
                        tag="Training/step_{}_{}".format(step, tag),
                    )
                    sampling_rate = preprocess_config["preprocessing"]["audio"][
                        "sampling_rate"
                    ]

                    log(
                        train_logger,
                        audio=wav_reconstruction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_reconstructed".format(step, tag),
                    )
                    log(
                        train_logger,
                        audio=wav_prediction,
                        sampling_rate=sampling_rate,
                        tag="Training/step_{}_{}_synthesized".format(step, tag),
                    )
                if step == total_step:
                    quit()
                    wandb.finish()
                step += 1
                outer_bar.update(1)
    wandb.finish()     

if __name__ == "__main__":
    print("main")

    main()
    