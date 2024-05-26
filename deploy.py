import streamlit as st
import os
import cv2
import time
import torch
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
from transformers import VisionEncoderDecoderModel,TrOCRProcessor
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from math import sin,pi
from arguments import parse_args
import tempfile

class PIYODA:
    def __init__(self, FLAGS):
        self.DEVICE = torch.device(FLAGS.device_name)
        self.MAIN_MODEL = YOLO(FLAGS.main_model)
        self.ZEBRA_MODEL = YOLO(FLAGS.zebra_model)
        self.LPD_MODEL = YOLO(FLAGS.lpd_model)
        self.LPR_MODEL = VisionEncoderDecoderModel.from_pretrained(FLAGS.lpr_model, local_files_only=True).to(self.DEVICE)
        self.PROCESSOR = TrOCRProcessor.from_pretrained(FLAGS.processor)
        self.stopline = None # stopline uzunligi pixel
        self.stopline_direction = None # stopline ning zebradan tepada yoki pastda bo'lishi yo'nalishi
        self.stopline_meter = FLAGS.stopline_meter # stop line uzunligi metr
        self.count_padestrian_for_violation = FLAGS.count_padestrian_for_violation # Qoidabuzarlik hisoblanishi uchun PO'Y da harakatlanayotgan piyodalar soni
        self.left_right_range_from_vehicle = FLAGS.left_right_range_from_vehicle # Piyoda transport Boxidan shu masofa uzoqlikda bo'lsa qoidabuzarlik hisoblanmaydi
        self.right_left_distance_from_vehicle = FLAGS.right_left_distance_from_vehicle # Piyoda transport Boxiga shu masofadan yaqinroq bo'lsa qoidabuzarlik hisoblanadi
        self.camera_angle = FLAGS.camera_angle
        self.two_meter_how_many_pixel = FLAGS.two_meter_how_many_pixel
        
        self.zebra_polygon = FLAGS.zebra_polygon
        self.road_polygon = FLAGS.road_polygon
        self.zebra_polygon_points = np.array(FLAGS.zebra_polygon_points, np.int32).reshape((-1, 1, 2))
        self.road_polygon_points = np.array(FLAGS.road_polygon_points, np.int32).reshape((-1, 1, 2))
        
    def STOPLINE_PARAMETERS(self):
        sinus = sin(self.camera_angle*pi/180.)
        x = self.two_meter_how_many_pixel/(1.+sinus)

        distance = 0
        
        if self.stopline_meter>=0:
            for metr in range(self.stopline_meter):
                distance += x*sinus**metr
            return int(distance), -1
            
        else:
            for metr in range(abs(self.stopline_meter)):
                distance += x/sinus**(metr+1)
            return int(distance), 1

    def OCR(self, vehicle):
        result = self.LPD_MODEL.predict(vehicle, conf=0.7, device = self.DEVICE, verbose=False)
        if len(result[0])==0:
            return ''
            
        xmin,ymin,xmax,ymax = result[0].boxes.xyxy.cpu().numpy().astype(int)[0]
        licence_plate = vehicle[ymin:ymax, xmin:xmax]
        
        pixel_values = self.PROCESSOR(licence_plate, return_tensors='pt').pixel_values.to(self.DEVICE)
        generated_ids = self.LPR_MODEL.generate(pixel_values)
        generated_text = self.PROCESSOR.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return generated_text

    def BOUNDING_BOX(self, frame, boxes, class_ids):
        color = (255,0,0)
        
        for box, class_id in zip(boxes, class_ids):
            if class_id in [2,5,7]:
                color = (0, 0, 255)
        
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), color, 2)
    
    def KRITERIY(self, boxes, track_ids, class_ids, track, stopline):
        people, padestrians, vehicles, motorbikes = [], [], [], []
        violator = defaultdict(lambda: [])
        
        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            if class_id in [2,5,7] and box[-1]>=stopline:
                vehicles.append([track_id, box])
            
            elif class_id==0 and self.zebra_polygon.contains(Point(int(box[0]+box[2])//2, int(box[1]+box[3])//2)):
                people.append([track_id, box])

            elif class_id==3:
                motorbikes.append([track_id, box])

        for padestrian in people:
            
            motorbiker = False
            real = True
            
            padestrian_track_id, padestrian_box = padestrian
            x_padestrian_min, y_padestrian_min, x_padestrian_max, y_padestrian_max = padestrian_box
            
            for motorbike in motorbikes:
                motorbike_track_id, motorbike_box = motorbike
                x_motorbike_min, y_motorbike_min, x_motorbike_max, y_motorbike_max = motorbike_box
                
            if motorbiker:
                continue
                
            for vehicle in vehicles:
                vehicle_track_id, vehicle_box = vehicle
                x_vehicle_min, y_vehicle_min, x_vehicle_max, y_vehicle_max = vehicle_box
                
                if x_vehicle_min-5<=x_padestrian_min and x_padestrian_max<=x_vehicle_max+5 and y_vehicle_min-5<=y_padestrian_min and y_padestrian_max<=y_vehicle_max+5:
                    real = False

            if real:
                padestrians.append([padestrian_track_id, padestrian_box])
        
        if len(vehicles) * len(padestrians) == 0:
            return None

        elif len(padestrians)>=self.count_padestrian_for_violation:
            
            for vehicle in vehicles:
                vehicle_track_id, vehicle_box = vehicle
                x_vehicle_min,_, x_vehicle_max,_ = vehicle_box
                
                for padestrian in padestrians:
                    padestrian_track_id, padestrian_box = padestrian
                    x_padestrian_min,_, x_padestrian_max,_ = padestrian_box

                    if len(track[padestrian_track_id])==4:
                        old,_,_,new = track[padestrian_track_id]

                        if int(new[0])-int(old[0])+5>0: # Harakat o'ngga
                            if x_vehicle_max+self.left_right_range_from_vehicle>x_padestrian_min and x_vehicle_min-self.right_left_distance_from_vehicle<x_padestrian_max:
                                violator[vehicle_track_id].append(padestrian_track_id)

                        else: # Harakat chapga
                            if x_vehicle_min-self.left_right_range_from_vehicle<x_padestrian_max and x_vehicle_max+self.right_left_distance_from_vehicle>x_padestrian_min:
                                violator[vehicle_track_id].append(padestrian_track_id)
        return violator
        
    def VISUALIZE_VIOLATOR(self, frame, boxes, track_ids, violator, LP):
        overlay = frame.copy()
        
        vehicles = list(violator.keys())
        padestrians = list(set([item for sublist in list(violator.values()) for item in sublist]))
        
        for padestrian in padestrians:
            index = list(track_ids).index(padestrian)
            xmin,ymin,xmax,ymax = boxes[index]
    
            cv2.rectangle(overlay, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255,255,51), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            
        for vehicle in vehicles:
            index = list(track_ids).index(vehicle)
            xmin,ymin,xmax,ymax = boxes[index]
            
            cv2.putText(frame, str(LP[track_id]), (int(xmin), int(ymax)), cv2.FONT_HERSHEY_SIMPLEX , 1, (255,255,255), 2, cv2.LINE_AA)
            cv2.rectangle(overlay, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (204,0,0), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        print(f"{padestrians} - {vehicles}")
        return frame

    def VISUALIZE(self, frame, boxes, track_ids, class_ids):
        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            color = (255, 255, 0)
            class_ = "Odam"
            
            if class_id in [2,5,7]: 
                color = (255, 0, 0)
                class_ = "Avtomobil"
                            
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(frame, f"{class_} ({track_id})", (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (255,255,255), 1, cv2.LINE_AA) 
            
    def run(self, source):
        print(source)
        cap = cv2.VideoCapture(source)
        FRAME_WINDOW = st.image([])
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        track_history = defaultdict(lambda: [])
        LP = defaultdict(str)
        
        zebra_left_xy, zebra_right_xy, stopline = None, None, None

        while cap.isOpened():
            start = time.time()
            ret, frame = cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
            if not ret:
                break
        
            results = self.MAIN_MODEL.track(frame, classes = [0,2,5,7], conf = 0.5, persist=True, device = self.DEVICE, verbose = False)
            
            if len(results[0])==0 and zebra_left_xy is None and zebra_right_xy is None:
                zebra = self.ZEBRA_MODEL.predict(frame, conf=0.7, device = self.DEVICE)
                zebra_boxes = zebra[0].boxes.xyxy.cpu().numpy()
            
                first_col = zebra_boxes[:,0]
                second_col = zebra_boxes[:,1]
            
                min_zebra_y = np.min(second_col).astype(int)
                max_zebra_y = np.max(second_col).astype(int)
                
                if second_col[0]>second_col[-1]:
                    zebra_left_xy = (0, min_zebra_y)
                    zebra_right_xy = (width, max_zebra_y)
                    
                else:
                    zebra_left_xy = (0, max_zebra_y)
                    zebra_right_xy = (width, min_zebra_y)

                self.stopline, self.stopline_direction = self.STOPLINE_PARAMETERS()
                    
            if zebra_left_xy is not None and zebra_right_xy is not None:
                cv2.line(frame, zebra_left_xy, zebra_right_xy, (255,0,0), 2)
                cv2.line(frame, (zebra_left_xy[0], zebra_left_xy[1]+self.stopline_direction*self.stopline), (zebra_right_xy[0], zebra_right_xy[1]+self.stopline_direction*self.stopline), (255,0,0), 2)
                
                stopline = int(zebra_left_xy[1]+zebra_right_xy[1]+2*self.stopline_direction*self.stopline)//2
                
            try:
                if len(results[0].boxes)>0:
                    boxes = results[0].boxes.xyxy.cpu()
                    track_ids = results[0].boxes.id.numpy().astype(int)
                    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
                            
                    for box, track_id, class_id in zip(boxes, track_ids, class_ids):
                        xmin, ymin, xmax, ymax = box.numpy().astype(int)
                            
                        if class_id in [2,5,7] and ymax>=stopline and LP[track_id]=='' and self.road_polygon.contains(Point(int(xmin+xmax)//2, int(ymin+ymax)//2)):
                            car = frame[ymin:ymax, xmin:xmax]
                            lp_text = self.OCR(car)
                            if len(lp_text)==8 or len(lp_text)==9:         
                                LP[track_id]=lp_text
                        
                        track = track_history[track_id]
                        track.append([xmin, ymin])
                                
                        if len(track) > 4:
                            track.pop(0)
    
                    self.VISUALIZE(frame, boxes, track_ids, class_ids)
                    violator = self.KRITERIY(boxes, track_ids, class_ids, track_history, stopline)
                    
                    if violator is not None and len(violator)>=1:
                        frame = self.VISUALIZE_VIOLATOR(frame, boxes, track_ids, violator, LP)
                        
            except:
                print("ishlavotti")
                pass

            cv2.polylines(frame, [self.road_polygon_points], True, (255,255,255), 2)
            cv2.polylines(frame, [self.zebra_polygon_points], True, (255,255,255), 2)                
            
            end = time.time()
            cv2.putText(frame, f"FPS: {round(1/(end-start),3)}", (50,50), cv2.FONT_HERSHEY_SIMPLEX , 1, (255,255,255), 2, cv2.LINE_AA) 
            FRAME_WINDOW.image(frame)  

def main():
    FLAGS = parse_args()
    with st.sidebar:
        st.title("Sozlamalar")
        device_name = st.selectbox("Qurilma:",("CPU", "GPU"), index=1)
        
        if device_name=="CPU":
            FLAGS.device_name = "cpu"

        version_default = FLAGS.main_model[-4]
        versions = ["n","s","m","l","x"]
        
        version = st.selectbox("Yolov8:",("n", "s", "m", "l", "x"), index=versions.index(version_default))
        FLAGS.main_model = f'yolov8{version}.pt'
        
        stopline_meter = st.text_input("Stopliniya masofasi: ", placeholder=str(FLAGS.stopline_meter))
        if len(stopline_meter)>0:
            FLAGS.stopline_meter = int(stopline_meter)

        rl_lr_distance = st.text_input("Piyoda avtomobil oldidan o'tib ketgandan keyin qoida buzilish inobatga olinmaydigan masofa: ", placeholder=str(FLAGS.left_right_range_from_vehicle))
        if len(rl_lr_distance)>0:
            FLAGS.left_right_range_from_vehicle = float(rl_lr_distance)

        rr_ll_distance = st.text_input("Piyodaga yo'l berilishi kerak bo'lgan masofa: ", placeholder=str(FLAGS.right_left_distance_from_vehicle))
        if len(rr_ll_distance)>0:
            FLAGS.right_left_distance_from_vehicle = float(rr_ll_distance)
        
        camera_angle = st.text_input("Kamera qiyalik burchagi: ", placeholder=str(FLAGS.camera_angle))
        if len(camera_angle)>0:
            FLAGS.camera_angle = float(camera_angle)

        two_meter = st.text_input("2 m masofaning kadrdagi pixel qiymati: ", placeholder=str(FLAGS.two_meter_how_many_pixel))
        if len(two_meter)>0:
            FLAGS.two_meter_how_many_pixel = int(two_meter)
    
    video = st.file_uploader("Choose a video...", type=["mp4", "mpeg"])
    
    if video is not None:
        piyoda = PIYODA(FLAGS)
        video_path = None
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            video_path = temp_file.name
            temp_file.write(video.read())
            piyoda.run(video_path)
        os.remove(video_path)
        
        
if __name__ == "__main__":
    main()