import cv2
import numpy as np
import dlib
import os
from mediapipe import solutions


# parameter
lip_box = 140
hand_box = 320

# lip predictor
detector = dlib.get_frontal_face_detector()
lip_predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

#hand predictor
hand_predictor = solutions.hands.Hands()

    
def get_lip_margin(frame_copy):
    frameGray = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2GRAY)
    faces = detector(frameGray)

    for face in faces:
        x1,y1 = face.left(),face.top()
        x2,y2 = face.right(),face.bottom()
        # imgOriginal = cv2.rectangle(img, (x1,y1),(x2,y2),(0,255,0),2)
        landmarks = lip_predictor(frameGray,face)
        myPoints =[]
        for n in range(68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            myPoints.append([x,y])

        myPoints = np.array(myPoints)
        # 嘴唇区域mask提取
        x,y,w,h = cv2.boundingRect(myPoints[48:61])
        return x, y, w, h


def get_hand_margin(frame_copy):
    frameRGB = cv2.cvtColor(frame_copy, cv2.COLOR_BGR2RGB)
    results = hand_predictor.process(frameRGB)

    if results.multi_hand_landmarks:
        landmark_list = []
        # print(results.multi_hand_landmarks)
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark_id, finger_axis in enumerate(hand_landmarks.landmark):
                #便利某个手的每个关节
                # print(finger_axis.x)
                h, w, c = frameRGB.shape
                landmark_list.append([finger_axis.x*w, finger_axis.y*h])
        myPoints = np.array(landmark_list)

        x,y,w,h = cv2.boundingRect(myPoints.astype(np.int))
        return x,y,w,h
    return None  


     
def segment(data_path):
    person_dirs = os.listdir(data_path)
    person_paths = []
    for person in person_dirs:
        if len(person) == 2 and person=='xp':
            person_paths.append(person)

    for p_dir in person_paths:
        person_path = os.path.join(data_path, p_dir)
        lip_save_path = os.path.join(data_path, 'segmented', p_dir, 'lip')
        hand_save_path = os.path.join(data_path, 'segmented', p_dir, 'hand')
        hand_pos_path = os.path.join(data_path, 'segmented', p_dir, 'position')
        if not os.path.isdir(lip_save_path):
            os.makedirs(lip_save_path)
        if not os.path.isdir(hand_save_path):
            os.makedirs(hand_save_path)
        if not os.path.isdir(hand_pos_path):
            os.makedirs(hand_pos_path)
        
        
        all_files = os.listdir(person_path)
        num_files = 0
        for name in all_files:
            ext = os.path.splitext(name)[1]
            pre = os.path.splitext(name)[0]
            idx = int(pre.split('-')[1])

            if ext == '.mp4' and (idx==883 or idx==851 or idx==848):
                num_files = num_files + 1
                video_path = person_path + '/' + pre + ".mp4"
                print(video_path)
            
                videoCapture = cv2.VideoCapture(video_path)

                # video information
                fps = int(round(videoCapture.get(cv2.CAP_PROP_FPS)))
                width = int(videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frame_counter = int(videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
                # print(height)
                # print(width)

                
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                lip_videoWriter = cv2.VideoWriter(lip_save_path + '/'+ pre + '.mp4', fourcc, fps, (lip_box, lip_box))
                hand_videoWriter = cv2.VideoWriter(hand_save_path + '/'+ pre + '.mp4', fourcc, fps, (hand_box, hand_box))
                hand_position = []
                
                frame_idx = 0
               
                while True:
                    success, frame = videoCapture.read()

                    frame_idx = frame_idx + 1
                    if success:
                        frame_copy = frame.copy()
                        lip_point_info = get_lip_margin(frame_copy)
                        hand_point_info = get_hand_margin(frame_copy)

                        if hand_point_info != None and lip_point_info !=None:
                            print(hand_point_info)
                            print(lip_point_info)
                            # lip segmentation
                            lip_x, lip_y, _, _ = lip_point_info
                            lip_y = lip_y - 40
                            lip_x = lip_x - 22
                            if lip_y < 0:
                                lip_y = 0
                            if lip_x < 0:
                                lip_x = 0

                            if (lip_y+lip_box) >= frame.shape[0]:
                                lip_y = lip_y-((lip_y+lip_box)-frame.shape[0])
                                if lip_y < 0:
                                    lip_y = 0

                            if (lip_x+lip_box) >= frame.shape[1]:
                                lip_x = lip_x-((lip_x+lip_box)-frame.shape[1])
                                if lip_x < 0:
                                    lip_x = 0

                            lip_frame = frame[lip_y:lip_y+lip_box,lip_x:lip_x+lip_box,:]

                            # hand segmentation
                            hand_x, hand_y, _, _ = hand_point_info
                            hand_y = hand_y - 40
                            hand_x = hand_x - 40
                            if hand_x < 0:
                                hand_x = 0
                            if hand_y < 0:
                                hand_y = 0
                            frame_copy = frame.copy()
                            if (hand_y+hand_box) > frame.shape[0]:
                                hand_y = hand_y-((hand_y+hand_box)-frame.shape[0])
                                if hand_y < 0:
                                    hand_y = 0
                            if (hand_x+hand_box) > frame.shape[1]:
                                hand_x = hand_x-((hand_x+hand_box)-frame.shape[1])
                                if hand_x < 0:
                                    hand_x = 0
                            hand_frame = frame_copy[hand_y:hand_y+hand_box,hand_x:hand_x+hand_box,:]
                            
                            # print(hand_frame.shape)
                            lip_videoWriter.write(lip_frame)
                            hand_videoWriter.write(hand_frame) 
                            hand_position.append([hand_x, hand_y])
                            
                    else:
                        videoCapture.release()
                        lip_videoWriter.release()
                        hand_videoWriter.release()
                        hand_position = np.array(hand_position)
                        np.save(hand_pos_path + '/'+ pre + '.npy', hand_position)

                        break
                
                print(video_path + ' is segmented!')

                
# def test(data_path):
#     person_dirs = os.listdir(data_path)
#     person_paths = []
#     for person in person_dirs:
#         if len(person) == 2:
#             person_paths.append(person)

#     error = 0
#     for p_dir in person_paths:
#         person_path = os.path.join(data_path, p_dir)
        
#         lip_save_path = os.path.join(data_path, 'segmented', p_dir, 'lip')
#         hand_save_path = os.path.join(data_path, 'segmented', p_dir, 'hand')
#         # print(lip_save_path)
        
        
#         all_files = os.listdir(person_path)
#         # print(len(all_files))
#         num_files = 0
#         for name in all_files:
#             ext = os.path.splitext(name)[1]
#             pre = os.path.splitext(name)[0]
#             if ext == '.mp4':
#                 num_files = num_files + 1
#                 lip_video_path = lip_save_path + '/' + pre + ".mp4"
#                 lip_videoCapture = cv2.VideoCapture(lip_video_path)
#                 hand_video_path = lip_save_path + '/' + pre + ".mp4"
#                 hand_videoCapture = cv2.VideoCapture(hand_video_path)

#                 # video information
#                 lip_frame_counter = int(lip_videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))
#                 hand_frame_counter = int(lip_videoCapture.get(cv2.CAP_PROP_FRAME_COUNT))

#                 print('lip_frame_counter',hand_frame_counter)
#                 print('hand_frame_counter',hand_frame_counter)
#                 if lip_frame_counter!=hand_frame_counter:
#                     error+=1
#                 else:
#                     pass
                    
    
#     print(error)
                    

                
data_path = '/mnt/mntnfs/chinese_cued_speech_data'
segment(data_path)
# test(data_path)


# lip = cv2.VideoCapture('/mntnfs/med_data4/cs/test/segmented/LF/lip/0001.mp4')
# frame_counter = int(lip.get(cv2.CAP_PROP_FRAME_COUNT))
# print(frame_counter)

                
# hand = cv2.VideoCapture('/mntnfs/med_data4/cs/test/segmented/LF/hand/0001.mp4')
# frame_counter = int(hand.get(cv2.CAP_PROP_FRAME_COUNT))
# print(frame_counter)
