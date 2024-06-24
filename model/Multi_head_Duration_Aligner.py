import os
import sys
sys.path.append('/workspace/CODE/Jenny_Test')
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import json
import copy
import math
from collections import OrderedDict
from torch.nn.utils import weight_norm
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from utils.tools import get_mask_from_lengths, pad
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from utils.tools import init_weights, get_padding
from transformer import Encoder, Lip_Encoder
LRELU_SLOPE = 0.1
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




class CTC_classifier_MDA(nn.Module):
    """
    CTC_classifier_MDA
    nn.Linear를 사용해 입력 feature를 num_classes차원으로 분류한다. 
    forward에서는 입력 텐서의 크기를 조정하고, Linear Classifier를 적용한 후 원래 크기로 변환한다.
    CTC 손실을 계산하기 위한 분류기 모듈이다. 
    입력을 받아서 phoneme분류를 수행한다. 
    텍스트와 비디오 컨텍스트 시퀀스 정렬을 돕는 보조적인 loss로 사용된다. 
    """
    def __init__(self, num_classes):
        super().__init__()

        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        # B, S, 512
        size = x.size()
        x = x.reshape(-1, size[2]).contiguous()
        x = self.classifier(x)
        return x.reshape(size[0], size[1], -1)  
    
class Multi_head_Duration_Aligner(nn.Module):
    """Multi_head_Duration_Aligner"""
    def __init__(self, model_config):
        super(Multi_head_Duration_Aligner, self).__init__()
        # self.dataset_name = preprocess_config["dataset"]
        self.encoder = Encoder(model_config)
        self.lip_encoder = Lip_Encoder(model_config)
        self.attn = nn.MultiheadAttention(256, 8, dropout=0.1)
        self.attn_text_spk = nn.MultiheadAttention(256, 8, dropout=0.1)
        
        self.num_upsamples = len(model_config["upsample_ConvTranspose"]["upsample_rates"])
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(model_config["upsample_ConvTranspose"]["upsample_rates"],
                                       model_config["upsample_ConvTranspose"]["upsample_kernel_sizes"])):
            self.ups.append(weight_norm(
                ConvTranspose1d(model_config["upsample_ConvTranspose"]["upsample_initial_channel"],
                                model_config["upsample_ConvTranspose"]["upsample_initial_channel"], k,
                                u, padding=(u // 2 + u % 2), output_padding=u % 2)))
               
        self.proj_con = nn.Conv1d(256, 256, kernel_size=1, padding=0, bias=False)
        
        self.CTC_classifier_MDA = CTC_classifier_MDA(model_config["Symbols"]["phonemenumber"])  # len(symbols)
        

    def forward(
            self,
            lip_embedding,
            lip_masks,
            texts,
            src_masks,
            max_src_len,
            lip_lens, 
            src_lens,
            reference_embedding = None,
    ):
        output_lip = self.lip_encoder(lip_embedding, lip_masks)
        output_text = self.encoder(texts, src_masks)
        
        # Before calculating attention between phoneme and lip-motion sequence, the text information will be fused with the speaker identity, following the paper.
        sss = reference_embedding.unsqueeze(1).expand(-1, max_src_len, -1)
        contextual_sss, _ = self.attn_text_spk(query=output_text.transpose(0, 1), key=sss.transpose(0, 1),
                                        value=sss.transpose(0, 1), key_padding_mask=src_masks)

        contextual_sss = contextual_sss.transpose(0,1)
        output_text = contextual_sss + output_text
        
        output, _ = self.attn(query=output_lip.transpose(0, 1), key=output_text.transpose(0, 1),
                                        value=output_text.transpose(0, 1), key_padding_mask=src_masks)
        output = output.transpose(0,1)
        
        output = self.proj_con(output.transpose(1, 2))

        # In our implementation, we use the CTC as an auxiliary loss to help the text-video context sequence aligning information, like the diagonal alignment constraint loss on the attention output matrix in NeuralDubber (https://tsinghua-mars-lab.github.io/NeuralDubber/).  
        B = texts.shape[0] 
        ctc_pred_MDA_video = self.CTC_classifier_MDA(output.transpose(1, 2))
        
        # video length to mel length
        for i in range(self.num_upsamples):
            output = F.leaky_relu(output, LRELU_SLOPE)
            output = self.ups[i](output)
        output = output.transpose(1, 2)

        return (output, ctc_pred_MDA_video)