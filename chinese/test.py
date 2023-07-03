import cv2
import os
import torch
import numpy as np
speaker =['wt', 'lf', 'hs', 'xp']

def test(data_path):
    sum_frame = 0
    person_dirs = os.listdir(data_path)
    person_paths = []
    print(person_dirs)
    for person in person_dirs:
        if len(person) == 2 and person in speaker:
            person_paths.append(person)

    # print(person_paths)
    for p_dir in person_paths:
        person_path = os.path.join(data_path, p_dir)
        lip_save_path = os.path.join(data_path, 'segmented', p_dir, 'lip')
        hand_save_path = os.path.join(data_path, 'segmented', p_dir, 'hand')
        
        hand_files = os.listdir(hand_save_path)
        lip_files = os.listdir(lip_save_path)
        # if p_dir == 'lf':
        #     print(lip_files)
        
        for lip, hand in zip(lip_files, hand_files):

            lip_video_path = lip_save_path + '/' + lip
            hand_video_path = hand_save_path + '/' + hand
            print(lip_video_path)


            lipCapture = cv2.VideoCapture(lip_video_path)
            handCapture = cv2.VideoCapture(hand_video_path)

            # video information
            lipFrame = int(lipCapture.get(cv2.CAP_PROP_FRAME_COUNT))
            # print(lipFrame)
            sum_frame += lipFrame
            # handFrame = int(handCapture.get(cv2.CAP_PROP_FRAME_COUNT))
            # print(lipFrame)
            # if lipFrame == 0:
            #     print(lip_video_path)
            # if handFrame == 0:
            #     print(hand_video_path)
    print(sum_frame)

# test('/mntnfs/med_data4/cs')

# a = [1,2,3]
# print(a[:-1])

# a = torch.randn((4,1, 3))
# print(a)
# print(a.shape)

# b = torch.argmax(a[:,-1,:], dim=-1, keepdim=True)
# print(b.shape)

a = np.array([12, 32])
b =np.array([12, 32])

print(a.tostring())
print(b.tostring())

# path = '/mntnfs/med_data4/cs/segmented/lf/lip/LF-0289.mp4'
# lipCapture = cv2.VideoCapture(path) 
# lipFrame = int(lipCapture.get(cv2.CAP_PROP_FRAME_COUNT))
# print(lipFrame)