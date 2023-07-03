import os
import cv2
import sys
sys.path.append("..")
from config import args

def is_frame_null(data_path, p_dir, name):
    lip_video_path = os.path.join(data_path, 'segmented', p_dir, 'lip', name)
    hand_video_path = os.path.join(data_path, 'segmented', p_dir, 'hand', name)
    lip_stream = cv2.VideoCapture(lip_video_path)
    hand_stream = cv2.VideoCapture(hand_video_path)
    lip_frame_count = int(lip_stream.get(cv2.CAP_PROP_FRAME_COUNT))
    hand_frame_count = int(hand_stream.get(cv2.CAP_PROP_FRAME_COUNT))

    if lip_frame_count == 0 or hand_frame_count==0:
        return True
    else:
        return False

# write person, file_name, label_text into a txt file
# less than train_idx for trianing, larger than train_idx for testing
def generate_txt(data_path, label_txt_path):
    # os.remove(label_txt_path + 'sentence_labels.txt')
    os.remove(label_txt_path + 'frame_labels.txt')
    sentence_label_txt = open(label_txt_path + 'frame_labels.txt', 'a')
    # test_label_txt = open(label_txt_path + 'test_labels.txt', 'a')
    person_dirs = os.listdir(data_path)

    person_paths = []
    for person in person_dirs:
        if len(person) == 2:
            person_paths.append(person)

    for p_dir in person_paths:
        ext = os.path.splitext(p_dir)[1]
        person_path = os.path.join(data_path, p_dir)
        all_files = os.listdir(person_path)
        for name in all_files:
            ext = os.path.splitext(name)[1]
            pre = os.path.splitext(name)[0]
            idx = int(pre.split('-')[1])
            if ext == '.mp4':
                sentence_label = get_full_label(person_path, pre)
                
                sentence_text = p_dir + ',' + name + ',' + sentence_label
                
                if is_frame_null(data_path, p_dir, name) is True:
                    pass
                else:
                    sentence_label_txt.write(sentence_text)
                    sentence_label_txt.write('\n')
               

yunmu = ['a', 'o', 'e', 'i', 'u', 'v', 'ai', 'ei', 'ao',
         'ou', 'er', 'an', 'en', 'ang', 'eng', 'ong']

# read textgrid file to get label text information
def get_label(person, name):
    grid_file = os.path.join(data_path, person, name+'-V.TextGrid')

    with open(grid_file, "r") as f:  # 打开文件
        lines = f.read()
        lines = lines.split('\n')  # 读取文件
        label = ''
        for line in lines:
            if 'text =' in line:
                sub_text = line.split('\"')[1]
                sub_text = sub_text.strip()

                if sub_text != '' and sub_text != 'noise':
                    if sub_text in yunmu:
                        sub_text = sub_text + ' ' + '-'
                    if label != '':
                        sub_text = ' ' + sub_text
                    label = label + sub_text
        
    # join_str=','
    # label_str = join_str.join(label)
    len_label = len(label)
    label = label[:len_label-2] # remove " -" at the end
    print('generate label:' + label)
    return label


# read *-V.TextGrid file and generate label for each frame
def get_full_label(person_path, file_name, fps=30):
    label_file = person_path + '/' + file_name + "-V.TextGrid"
    with open(label_file, "r") as f:  # 打开文件
        lines = f.read()
        lines = lines.split('\n')  # 读取文件

        stamp_start = []
        stamp_end = []
        text_seq = []
        for line in lines:
            if 'intervals: size' in line:
                num_intervals = int(line.split(' ')[-1])
            elif 'xmin' in line:
                stamp_start.append(float(line.split(' ')[-1]))
            elif 'xmax' in line:
                stamp_end.append(float(line.split(' ')[-1]))
            elif 'text =' in line:
                if line.split('\"')[1] == '':
                    text_seq.append('-')
                else:
                    sub_text = line.split('\"')[1]
                    sub_text = sub_text.strip()
                    text_seq.append(sub_text)
            else:
                pass

    if (len(stamp_start) - len(text_seq)) != 2 or (len(stamp_end) - len(text_seq)) != 2 or len(text_seq) != num_intervals:
        print(label_file)
        return None

    full_label = ''
    min_len = min(len(stamp_start[2:]), len(text_seq), len(stamp_end[2:]), num_intervals)
    stamp_start, stamp_end = stamp_start[2:min_len+2], stamp_end[2:min_len+2]
    text_seq = text_seq[:min_len]

    for xmin, xmax, sub_label in zip(stamp_start, stamp_end, text_seq):
        frame_start = int(xmin * fps)
        frame_end = int(xmax * fps)
        if full_label != '':
            sub_label = ' ' + sub_label
        else:
            full_label = full_label + sub_label
            sub_label = ' ' + sub_label
            frame_end = frame_end -1

        for idx in range(frame_end - frame_start):
            full_label = full_label + sub_label
    # if len(full_label) < frame_counter:
    #     for idx in range((frame_counter - len(full_label))):
    #         full_label.append(sub_label)
    # elif len(full_label) > frame_counter:
    #     full_label = full_label[:frame_counter]
    # else:
    #     pass
    return full_label


def get_time_stamp(path, name):
    start_times = []
    end_times = []
    flags = []
    file = path + '/' + name
    with open(file, "r") as f:  # 打开文件
        data = f.read()
        lines = data.split('\n')  # 读取文件
        for line in lines:
            # if 'intervals' in line and 'size' in line:
            #     num_intervals = int(line.split(' ')[-1])
            if 'xmin = ' in line:
                start = float(line.strip().split(' ')[-1])
                start_times.append(start)
            elif 'xmax =' in line:
                end = float(line.strip().split(' ')[-1])
                end_times.append(end)
            elif 'text =' in line:
                if len(line.strip()) > 9:
                    flags.append(1)
                else:
                    flags.append(0)

    start_stamps = []
    end_stampls = []
    for xmin, xmax, flag in zip(start_times[1:], end_times[1:], flags):
        if flag == 1:
            start_stamps.append(xmin)
            end_stampls.append(xmax)
    return start_stamps, end_stampls






# def rename(data_path):
#     person_dirs = os.listdir(data_path)
#     person_paths = []
#     for person in person_dirs:
#         if len(person) == 2 and person=='xp':
#             person_paths.append(person)
#
#     for p_dir in person_paths:
#         person_path = os.path.join(data_path, p_dir)
#         all_files = os.listdir(person_path)
#         for name in all_files:
#             os.rename(person_path + '/' + name, person_path + '/' + "XP-"+name)
        



yunmu = ['an', 'e', 'o', 'a', 'ou', 'er', 'en', 'i', 'v',
         'ang', 'ai', 'u', 'ao', 'eng', 'ong', 'ei']

data_path = args['data_path']+'/'
label_txt_path = args['data_path']+'/'
# rename(data_path)
generate_txt(data_path, label_txt_path)


# seg_video(data_path)
# sample_video(data_path, ratio=1)

