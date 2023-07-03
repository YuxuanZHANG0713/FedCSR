import os
from pickle import TRUE
import random
import numpy as np
import cv2
from tqdm import tqdm
from PIL import Image

import sys
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from chinese.chi_config import chi_args
from chinese.autoaugment import CIFAR10Policy, Cutout

class Normalize(object):
    """
    will normalize each channel of the torch.*Tensor, i.e.
    channel = channel/127.5 - 1
    """

    def __call__(self, tensor):
        # TODO: make efficient
        for t in tensor:
            t.div_(0.5).sub_(1)
        return tensor

class ToTensor(object):

    def __call__(self, inputs):
        # handle numpy array
        outputs = torch.from_numpy(inputs).float()
        # backard compability
        
        return outputs


class CCSDataset(Dataset):

    # def __init__(self, data_path="/mntnfs/med_data4/cs", mode='segmented', is_train=True, phone2index=None, single=True):
    def __init__(self, data_path="/mntnfs/med_data4/cs", mode='segmented', is_train=True, phone2index=None, client='hs'):
      
        self.is_train = is_train
        if self.is_train:
            self.label_txt = data_path + "/fedlabels/" + client + "_train_labels.txt"    
            self.frame_label_txt = data_path + "/fedlabels/" + client + "_train_frame_labels.txt"    
        else:
            self.label_txt = data_path + "/fedlabels/" + client + "_test_labels.txt"    
            self.frame_label_txt = data_path + "/fedlabels/" + client + "_test_frame_labels.txt"    

        # data path: lip, hand
        self.data_path = data_path + '/' + mode
        # dict of phoneme
        self.phone2index = phone2index
        # sub_path(segmented, subsampling)
        self.mode = mode

        # file path of lip/hand, label
        self.lips, self.hands, self.hand_pos, self.labels, self.frame_labels = [],[],[],[],[]

        #record persons
        self.persons = []
        
        print("Load information from text: ", self.label_txt)
        print("Load data from folder: ", self.data_path)

        # adding file path and preprocess label
        max_len = 0
        min_len = 100000
        with open(self.label_txt, "r") as f:  # 打开文件
            data = f.read()
            lines = data.split('\n')  # 读取文件

            frame_f = open(self.frame_label_txt, "r")
            frame_data = frame_f.read()
            frame_lines = frame_data.split('\n')

            for line, frame_line in zip(lines,frame_lines):
                if len(line.split(',')) != 3 or len(frame_line.split(',')) != 3:
                    continue

                person, file_name, label = line.split(',')
                _, frame_file_name, frame_label = frame_line.split(',')
                if frame_file_name!=file_name:
                    print("index error")
                    exit()

                
                self.lips.append(os.path.join(self.data_path, person, 'lip', file_name))
                self.hands.append(os.path.join(self.data_path, person, 'hand', file_name))
                pos = np.load(os.path.join(self.data_path, person, 'position', file_name.replace('mp4', 'npy')), allow_pickle=True)
                self.hand_pos.append(pos.reshape((-1, 1, 1, 2)))
                
                label = label.split(' ')
                frame_label = frame_label.split(' ')
                if len(label)>max_len:
                    max_len = len(label)
                if len(label)<min_len:
                    min_len = len(label)
                self.labels.append(label)
                self.persons.append(person)
                self.frame_labels.append(frame_label)
            f.close()


        int_labels = []
        frame_init_labels = []
        for str_label, str_frame_label in zip(self.labels, self.frame_labels):
            # print(str_label)
            int_label = [self.phone2index[phone] for phone in str_label]
            int_label.append(self.phone2index["<EOS>"])
            frame_init_label = [self.phone2index[phone] for phone in str_frame_label]
            int_labels.append(int_label)
            
            frame_init_labels.append(frame_init_label)
            
            
        self.labels = int_labels
        self.frame_labels = frame_init_labels

        self.phone_code = [i for i in range(len(self.phone2index)+1)]

        print("max len of target", max_len)
        print("min len of target", min_len)

       
        # random shuffle
        random_list = list(zip(self.lips, self.hands, self.hand_pos, self.labels, self.frame_labels, self.persons))
        random.shuffle(random_list)
        self.lips[:], self.hands[:], self.hand_pos[:], self.labels[:], self.frame_labels[:], self.persons[:] = zip(*random_list)
        
        
    def __getitem__(self, index):
        # loading lip and hand videos.
        lip, hand, position, index = self.load_lip_hand(self.lips[index], self.hands[index], self.hand_pos[index], index) #(channel, frame, width, height)
       
       
        label = self.labels[index]
        frame_label = self.frame_labels[index]
        
        label = np.array(label)
        label_len = len(label)
        frame_label = np.array(frame_label)

        code_book = np.array(self.phone_code)
        inputCode = torch.from_numpy(code_book)

        # frame count
        lip_len = lip.shape[0]
        lip_len = torch.tensor(lip_len)
        
        # final target information
        inputTarget = torch.from_numpy(label)
        targetLen = torch.tensor(label_len) # length of a sentence
        frameTarget = torch.from_numpy(frame_label)

        # final data information
        inputData = (lip, hand, position)
        inputLen = lip_len

        return inputData, inputTarget, frameTarget, inputCode, inputLen, targetLen

    def __len__(self):
        return len(self.lips)


    def load_lip_hand(self, lip_path, hand_path, hand_pos, index):
        # initialize a VideoCapture object to read video data into a numpy array
        if self.is_train:
            self.transform = transforms.Compose([
                transforms.Resize([chi_args["roi_size"], chi_args["roi_size"]]),
                transforms.RandomCrop(chi_args["roi_size"], padding=16),
                CIFAR10Policy(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                #transforms.Normalize(mean=[0.626, 0.499, 0.449], std=[0.201, 0.219, 0.219])
                ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize([chi_args["roi_size"], chi_args["roi_size"]]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                #transforms.Normalize(mean=[0.626, 0.499, 0.449], std=[0.201, 0.219, 0.219])
                ])

        self.pos_transform = transforms.Compose([
                ToTensor(),
                Normalize()
            ])

        try:
            lip_stream = cv2.VideoCapture(lip_path)
            hand_stream = cv2.VideoCapture(hand_path)
            lip_frame_count = int(lip_stream.get(cv2.CAP_PROP_FRAME_COUNT))
            hand_frame_count = int(hand_stream.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(lip_stream.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(lip_stream.get(cv2.CAP_PROP_FRAME_HEIGHT))
            assert lip_frame_count <= 1000 and lip_frame_count > 0
        except (RuntimeError, AssertionError):
            lip_frame_count = 0
            while(lip_frame_count<=0 or lip_frame_count > 1000):
                index = np.random.randint(self.__len__())
                lip_stream = cv2.VideoCapture(self.lips[index])
                hand_stream = cv2.VideoCapture(self.hands[index])
                lip_frame_count = int(lip_stream.get(cv2.CAP_PROP_FRAME_COUNT))
                hand_frame_count = int(hand_stream.get(cv2.CAP_PROP_FRAME_COUNT))

        # load lip video and process it
        count, sample_count, retaining = 0, 0, True
        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        lip_buffer = np.empty((lip_frame_count, 3, chi_args["roi_size"], chi_args["roi_size"]), np.dtype('float16'))
        
        while (count <= lip_frame_count and retaining):
            retaining, lip_frame = lip_stream.read()
            if retaining:
                try:
                    image=cv2.cvtColor(lip_frame, cv2.COLOR_BGR2RGB)
                    image = Image.fromarray(image)
                    lip_buffer[sample_count] = self.transform(image)
                    # print(lip_buffer[sample_count])
                    sample_count += 1
                except cv2.error as err:
                    continue
            else:
                lip_stream.release()
                break
            count += 1
        lip_buffer = lip_buffer[:sample_count,:,:,:]


        # load hand video and process it
        count, sample_count, retaining = 0, 0, True
        # create a buffer. Must have dtype float, so it gets converted to a FloatTensor by Pytorch later
        hand_buffer = np.empty((hand_frame_count, 3, chi_args["roi_size"], chi_args["roi_size"]), np.dtype('float16'))
        pos_buffer = np.empty((hand_frame_count, 1, 1, 2), np.dtype('float16'))
        

        while (count <= hand_frame_count and retaining):
            retaining, hand_frame = hand_stream.read()
            if retaining:
                try:
                    hand_buffer[sample_count] = self.transform(Image.fromarray(cv2.cvtColor(hand_frame, cv2.COLOR_BGR2RGB)))
                    pos_buffer[sample_count] = self.pos_transform(hand_pos[count].astype(np.float16))
                    
                    sample_count += 1
                except Exception:
                    continue
            else:
                hand_stream.release()
                break
            count += 1
        hand_buffer = hand_buffer[:sample_count,:,:,:]

        lip_buffer = torch.from_numpy(lip_buffer)
        hand_buffer = torch.from_numpy(hand_buffer)
        pos_buffer = torch.from_numpy(pos_buffer)

        # make sure that lip and hand have the same frame count
        if lip_buffer.shape[0]!=hand_buffer.shape[0]:
            min_len = lip_buffer.shape[0] if lip_buffer.shape[0]< hand_buffer.shape[0] else hand_buffer.shape[0]
            lip_buffer, hand_buffer, pos_buffer = lip_buffer[:min_len,:,:,:], hand_buffer[:min_len,:,:,:], pos_buffer[:min_len,:,:,:]

        return lip_buffer, hand_buffer, pos_buffer, index

class LingDataset(Dataset):
    def __init__(self, phone2index, file_path):
        with open(file_path, 'r') as f:
            self.lines = f.readlines()
        self.phone2index = phone2index

    def __getitem__(self, index):
        line = self.lines[index]
        sentence = line.split(',')[2].strip('\n')
        return self.to_code(sentence)


    def __len__(self):
        return len(self.lines)

    def to_code(self, sentence):
        list_phone = sentence.split(' ')
        list_code = [self.phone2index[i] for i in list_phone]
        return torch.tensor(list_code)


# if __name__ == '__main__':
    
    # data_path = "\\mntnfs\\med_data5\\CS_data"

    # lenght = 0
    
    # trainData = CCSDataset(chi_args["data_path"], is_train=True, phone2index=chi_args["phone_to_index"], client='hs')
    # trainLoader = DataLoader(trainData, batch_size=chi_args["batch_size"], shuffle=True, drop_last=True)
    # for batch, (inputBatch, targetBatch, frameTarget, inputCode, inputLenBatch, targetLenBatch) in enumerate(trainLoader):
    #     inputLenBatch = inputLenBatch >> 2
    #     print(inputLenBatch-targetLenBatch)

    # file_path = 'C:\\Users\\dell\\Desktop\\SRIBD\\CuedSpeech\\dataset\\CS_data_FL\\fedlabels\\hs_train_labels.txt'

    # phone2index = {"b":1, "p":2, "m":3, "f":4, "d":5, "t":6, "n":7, "l":8, "g":9, "k":10, "h":11, "j":12,
    #                      "q":13, "x":14, "zh":15, "ch":16, "sh":17, "r":18, "z":19, "c":20, "s":21, "y":22, "w":23, "yu":24,
    #                      "a":25, "o":26, "e":27, "i":28, "u":29, "v":30, "ai":31, "ei":32, "ao":33, "ou":34, "er":35, "an":36,
    #                      "en":37, "ang":38, "eng":39, "ong":40, "-": 41, "<EOS>": 42}

    # lingData = LingDataset(phone2index=phone2index, file_path=file_path)
    # for i in lingData:
    #     print(i)

   
   