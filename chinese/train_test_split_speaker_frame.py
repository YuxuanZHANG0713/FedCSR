
import random
import sys
import os
sys.path.append("..")
from config import args
#ratio: 4:1


def split_label(label_txt, training_txt, test_txt, speaker_list):
    f = open(label_txt, "r")  # 打开文件
    data = f.read()
    lines = data.split('\n')  # 读取文件
    random.shuffle(lines)
    

    for speaker_item in speaker_list:
        speaker_lines =[]

        for line in lines:
            if len(line.split(',')) != 3:
                continue
            else:
                speaker, _, _ = line.split(',')
                if speaker == speaker_item:
                    speaker_lines.append(line)


        unit = int(len(speaker_lines)/5)
        train_lines = speaker_lines[:unit*4]
        test_lines = speaker_lines[unit*4:]

        label_txt = open(training_txt, 'a')
        
        for line in train_lines:
            if len(line.split(',')) != 3:
                continue
            label_txt.write(line)
            label_txt.write('\n')
            
        label_txt.close()
        

        label_txt = open(test_txt, 'a')
        
        for line in test_lines:
            if len(line.split(',')) != 3:
                continue
            label_txt.write(line)
            label_txt.write('\n')
            
        label_txt.close()

    f.close()


label_txt = args['data_path']+'/frame_labels.txt'
single_training_labels = args['data_path'] + '/single_speaker_train_labels_frame.txt'
single_test_labels = args['data_path'] +'/single_speaker_test_labels.txt_frame'
single_speaker_list = ['hs']

# split training set and test set for multiple speakers
multi_training_labels = args['data_path'] + '/multi_speaker_train_labels_frame.txt'
multi_test_labels = args['data_path'] + '/multi_speaker_test_labels_frame.txt'
multi_speaker_list = ['hs','lf', 'wt', 'xp']

# os.remove(single_training_labels)
# os.remove(single_test_labels)
# os.remove(multi_training_labels)
# os.remove(multi_test_labels)

split_label(label_txt, single_training_labels, single_test_labels, single_speaker_list)
split_label(label_txt, multi_training_labels, multi_test_labels, multi_speaker_list)