import json
import math
import os
import random
import numpy as np
from torch.utils.data import Dataset
import sys

sys.path.append("/workspace/CODE/Jenny_Test/model")
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from text import text_to_sequence
from utils.tools import pad_1D, pad_2D

class Dataset(Dataset):
    """
    1. __init__(), __len__(), __getitem__()을 구현한다. 
    - __init__() 초기화 및 필요 변수 설정 
    - __len__() 총 샘플 수 반환 
    - __getitem__() 주어진 인덱스에 해당하는 샘플 반환, 데이터 전처리 수행 가능 
    2. 데이터셋 인스턴스 생성
    - 필요한경우, train, val 셋으로 분할 가능
    3. DataLoader 인스턴스 생성 
    - 배치단위로 로드할 수 있는 데이터 로더 생성
    - DataLoader 주요 매개 변수 
    -- dataset : 데이터셋 인스턴스 
    -- batch_size : 배치 크기 
    -- suffle : 에포크마다 데이터셋을 섞을지 여부
    -- num_workers : 데이터 로드에 사용할 프로세스 수 
    -- collate_fn : 샘플을 배치로 묶을 때 사용하는 함수-optional
    """
    """
    1. Phoneme Feature, Speaker Feature, Lip-M Feature를 넣어서 Mel Generator -> Vocoder 수행
    2. Phoneme Feature, Speaker Feature 묶기
    3. Lip-M Feature 
    """
    def __init__(self,txt_file_name,preprocess_config,batch_size):
        #
        #txt_file_name = basename|speaker|text|raw_text로 구성된 파일

        self.preprocessed_path="/workspace/DATA/chem/preprocessed_3"
        #lists-name, speaker, text, raw_text
        self.basenames, self.speakers, self.texts, self.raw_texts = self.process_meta(txt_file_name)
        self.cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]
        #wandb로 뺄 것
        self.batch_size = batch_size
        self.sort = None
        self.drop_last = None

    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self,idx):
        basename = self.basenames[idx]
        speaker = self.speakers[idx]
        speaker = 'chem'
        speaker_id = 0
        # 오디오 데이터 길이: lip_embedding길이/16000
        # spk_path = os.path.join(
        #     self.preprocessed_path,
        #     "spk",
        #     "{}-{}-spk.npy".format(speaker, basename),
        #     )
        # spk = np.load(spk_path)
        
        raw_text = self.raw_texts[idx]
        phone = np.array(text_to_sequence(self.texts[idx], self.cleaners))
        mel_path = os.path.join(
            self.preprocessed_path,
            "mel",
            "{}-{}-mel.npy".format(speaker, basename),
        )
        mel = np.load(mel_path)
        mel = mel.T

        pitch_path = os.path.join(
            self.preprocessed_path,
            "pitch",
            "{}-{}-pitch.npy".format(speaker, basename),
        )
        pitch = np.load(pitch_path)
        energy_path = os.path.join(
            self.preprocessed_path,
            "energy",
            "{}-{}-energy.npy".format(speaker, basename),
        )
        energy = np.load(energy_path)
        duration_path = os.path.join(
            self.preprocessed_path,
            "duration",
            "{}-{}-duration.npy".format(speaker, basename),
        )
        duration = np.load(duration_path)

        #
        # feature_256_path = os.path.join(
        #     self.preprocessed_path,
        #     "VA_feature",
        #     "{}-feature-{}.npy".format(speaker, basename),
        # )
        # feature_256 = np.load(feature_256_path)

        #chem-face-_7s29Q76st0_face
        lip_embedding = None
        lip_embedding_path = os.path.join(
            self.preprocessed_path,
            # "extrated_embedding_Chem_gray",
            "lip",
            "{}-{}-lip.npy".format('chem', basename),
        )

        # lip_embedding = np.expand_dims(np.load(lip_embedding_path)['arr_0'],axis=0)
        lip_embedding = np.expand_dims(np.load(lip_embedding_path),axis=0)
        sample = {
        "id": basename,
        "raw_text": raw_text,
        "speaker": speaker_id,
        "text": phone,
        
        "mel": mel,
        "pitch": pitch,
        "energy": energy,
        "duration": duration,
        # "spk": spk,
        "lip_embedding": lip_embedding,
        }
        return sample 

    def reprocess(self, data, idxs):
        """
        데이터셋에서 샘플들을 선택하고 선택된 샘플들을 전처리하는 역할
        패딩(padding)을 적용하여 시퀀스 길이를 맞추거나, NumPy 배열로 변환하는 등의 작업
        전처리된 필드들을 튜플 형태로 반환
        """
        try:
            ids = [data[idx]["id"] for idx in idxs]
            speakers = [data[idx]["speaker"] for idx in idxs]
            texts = [data[idx]["text"] for idx in idxs]
            raw_texts = [data[idx]["raw_text"] for idx in idxs]
            mels = [data[idx]["mel"] for idx in idxs]
            pitches = [data[idx]["pitch"] for idx in idxs]
            energies = [data[idx]["energy"] for idx in idxs]
            durations = [data[idx]["duration"] for idx in idxs]
            #
            # spks = [data[idx]["spk"] for idx in idxs]
            # emotions = [data[idx]["emotion"] for idx in idxs]
            # emos = [data[idx]["emo"] for idx in idxs]
            # feature_256 = [data[idx]["feature_256"] for idx in idxs]

            lip_embedding = [data[idx]["lip_embedding"] for idx in idxs]

            text_lens = np.array([text.shape[0] for text in texts])#16*(3024,)
            mel_lens = np.array([mel.shape[0] for mel in mels]) #16*(18350,80)
            
            lip_lens = np.array([lip_e.shape[1] for lip_e in lip_embedding])#16*(1,6711,512)
            
                
            lip_embedding = [lip_e[0] for lip_e in lip_embedding]
            if max(lip_lens) * 4> max(mel_lens):
                print("Lip length is longer than mel length")
                
            speakers = np.array(speakers) # 16
            texts = pad_1D(texts) # 16x273
            mels = pad_2D(mels) # 16x2379x80
            pitches = pad_1D(pitches) # 16x273
            energies = pad_1D(energies) # 16x273
            durations = pad_1D(durations) # 16x273 Beaca
            # Since we don't need to use Length Regulator, convert word length to mel-spectrum length
            durations = np.array(durations)  # 16x273
            # spks = np.array(spks) # 16x256
            # emotions = np.array(emotions) # 16
            # emos = np.array(emos) # 16x256
            # feature_256 = pad_2D(feature_256)
            
            lip_embedding = pad_2D(lip_embedding,max(lip_lens))

            return (
                ids,
                raw_texts,
                speakers,
                texts,
                text_lens,
                max(text_lens),
                mels,
                mel_lens,
                max(mel_lens),
                pitches,
                energies,
                durations,
                # spks,
                # emotions,
                # emos,
                # feature_256,
                lip_lens,
                max(lip_lens),
                lip_embedding,
            )
        except Exception as e:
            print(e)
            print("Error occured in reprocess")
            return None 
    def collate_fn(self,data):
        """
        데이터 로더에 batch를 구성할때 호출되는 함수
        데이터로더는 샘플들을 가져와 배치단위로 그룹화함-> 배치내 샘플들은 동일한 크기를 가져야함
        배치내 샘플들을 전처리하고 배치단위로 반환
        preprocess 메서드를 사용해서 전처리함
        배치단위로 forward propagation -> backpropagation으로 파라미터 업데이트 
        """
        data_size = len(data)

        if self.sort:
            len_arr = np.array([d["text"].shape[0] for d in data])
            idx_arr = np.argsort(-len_arr)
        else:
            idx_arr = np.arange(data_size)

        tail = idx_arr[len(idx_arr) - (len(idx_arr) % self.batch_size) :]
        idx_arr = idx_arr[: len(idx_arr) - (len(idx_arr) % self.batch_size)]
        idx_arr = idx_arr.reshape((-1, self.batch_size)).tolist()
        if not self.drop_last and len(tail) > 0:
            idx_arr += [tail.tolist()]

        output = list()
        for idx in idx_arr:
            result = self.reprocess(data, idx)
            if result is not None:
                output.append(result)
            else:
                print("Error occured in collate_fn")

        return output
    def process_meta(self, filename):
            """
            metadata 
            데이터셋에 대한 부가적인 정보 포함
            -샘플 고유 식별자,샘플 파일 경로, 파일명 , 레이블, 추가데이터 등
            """
            with open(
                os.path.join(self.preprocessed_path, filename), "r", encoding="utf-8"
            ) as f:
                name = []
                speaker = []
                text = []
                raw_text = []
                for line in f.readlines():
                    n, s, t, r = line.strip("\n").split("|")
                    name.append(n)
                    speaker.append(s)
                    text.append(t)
                    raw_text.append(r)
                return name, speaker, text, raw_text

if __name__=="__main__":
    print("DataLoader Test")
    import torch
    import yaml 
    from torch.utils.data import DataLoader
    from utils.tools import to_device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    #txt_file_name,preprocess_config,batch_size
    preprocess_config = yaml.load(
        open("../config/Chem/preprocess.yaml", "r"), Loader=yaml.FullLoader
    )
    train_dataset = Dataset(
        "train.txt",
        preprocess_config,
        batch_size=16,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=16,
        shuffle=True,      
        collate_fn=train_dataset.collate_fn,
    )

    n_batch = 0
    for batchs in train_loader:
        for batch in batchs:
            to_device(batch, device)
            n_batch += 1
            
    #Training set  with size 230 is composed of 15 batches.
    print(
        "Training set  with size {} is composed of {} batches.".format(
            len(train_dataset), n_batch
        ))