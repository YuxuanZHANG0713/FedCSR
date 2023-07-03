
import random
import sys
import os
from chi_config import chi_args
#ratio: 4:1


def split_label(label_txt, frame_label_txt, training_txt, test_txt, frame_training_txt, frame_test_txt, speaker_list):
    f = open(label_txt, "r")  # 打开文件
    data = f.read()
    lines = data.split('\n')  # 读取文件
    lines.sort()

    f_frame = open(frame_label_txt, "r")  # 打开文件
    data_frame = f_frame.read()
    lines_frame = data_frame.split('\n')  # 读取文件
    lines_frame.sort()

    random_list = list(zip(lines, lines_frame))
    random.shuffle(random_list)
    lines[:], lines_frame[:] = zip(*random_list)

    speaker_mode = speaker_list[0]
    # for speaker_item in speaker_list:
    speaker_lines = []
    speaker_frame_lines = []

    for line, frame_line in zip(lines,lines_frame) :
        if len(line.split(',')) != 3 or len(frame_line.split(',')) != 3:
            continue
        else:
            speaker, _, _ = line.split(',')
            frame_speaker, frame_video, _ = frame_line.split(',')
                
            if speaker == speaker_mode and frame_speaker==speaker_mode:
                speaker_lines.append(line)
                speaker_frame_lines.append(frame_line)
           

    unit = int(len(speaker_lines)/5)
    train_lines = speaker_lines[:unit*4]
    test_lines = speaker_lines[unit*4:]

    train_frame_lines = speaker_frame_lines[:unit*4]
    test_frame_lines = speaker_frame_lines[unit*4:]


    label_txt = open(training_txt, 'a')
    frame_label_txt = open(frame_training_txt, 'a')
    
    for line, frame_line in zip(train_lines, train_frame_lines):
        
        label_txt.write(line)
        label_txt.write('\n')

        frame_label_txt.write(frame_line)
        frame_label_txt.write('\n')

        if len(speaker_list)>1:
            for other_spearker in speaker_list[1:]:
                other_line = line.replace(speaker_mode, other_spearker)
                other_line = other_line.replace(speaker_mode.upper(), other_spearker.upper())
                label_txt.write(other_line)
                label_txt.write('\n')

                other_frame_line = frame_line.replace(speaker_mode, other_spearker)
                other_frame_line = other_frame_line.replace(speaker_mode.upper(), other_spearker.upper())
                frame_label_txt.write(other_frame_line)
                frame_label_txt.write('\n')

    label_txt.close()
    frame_label_txt.close()
    

    label_txt = open(test_txt, 'a')
    frame_label_txt = open(frame_test_txt, 'a')
    
    
    for line, frame_line in zip(test_lines,test_frame_lines):
        
        label_txt.write(line)
        label_txt.write('\n')
        frame_label_txt.write(frame_line)
        frame_label_txt.write('\n')

        if len(speaker_list)>1:
            for other_spearker in speaker_list[1:]:
                other_line = line.replace(speaker_mode, other_spearker)
                other_line = other_line.replace(speaker_mode.upper(), other_spearker.upper())
                label_txt.write(other_line)
                label_txt.write('\n')

                other_frame_line = frame_line.replace(speaker_mode, other_spearker)
                other_frame_line = other_frame_line.replace(speaker_mode.upper(), other_spearker.upper())
                frame_label_txt.write(other_frame_line)
                frame_label_txt.write('\n')
        
    label_txt.close()
    frame_label_txt.close()

    f.close()

label_txt = chi_args['data_path']+'/sentence_labels.txt'
frame_label_txt = chi_args['data_path']+'/frame_labels.txt'

single_training_labels = chi_args['data_path'] + '/single_speaker_train_labels.txt'
single_test_labels = chi_args['data_path'] +'/single_speaker_test_labels.txt'
single_training_frame_labels = chi_args['data_path'] + '/single_speaker_train_frame_labels.txt'
single_test_frame_labels = chi_args['data_path'] +'/single_speaker_test_frame_labels.txt'
single_speaker_list = ['hs']

# split training set and test set for multiple speakers
multi_training_labels = chi_args['data_path'] + '/multi_speaker_train_labels.txt'
multi_test_labels = chi_args['data_path'] + '/multi_speaker_test_labels.txt'
multi_training_frame_labels = chi_args['data_path'] + '/multi_speaker_train_frame_labels.txt'
multi_test_frame_labels = chi_args['data_path'] + '/multi_speaker_test_frame_labels.txt'
multi_speaker_list = ['hs','lf', 'wt', 'xp']

os.remove(single_training_labels)
os.remove(single_test_labels)
os.remove(multi_training_labels)
os.remove(multi_test_labels)

os.remove(single_training_frame_labels)
os.remove(single_test_frame_labels)
# os.remove(multi_training_frame_labels)
# os.remove(multi_test_frame_labels)

split_label(label_txt, frame_label_txt, single_training_labels, single_test_labels, single_training_frame_labels, single_test_frame_labels, single_speaker_list)
split_label(label_txt, frame_label_txt, multi_training_labels, multi_test_labels, multi_training_frame_labels, multi_test_frame_labels, multi_speaker_list)