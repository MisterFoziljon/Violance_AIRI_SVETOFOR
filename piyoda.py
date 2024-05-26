import os
import cv2
import time
import torch
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
from transformers import (VisionEncoderDecoderModel,TrOCRProcessor)
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from math import sin,pi

class PIYODA:
    def __init__(self):
        self.DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.MAIN_MODEL = YOLO("models/yolov8x.pt")
        self.ZEBRA_MODEL = YOLO("models/zebra.pt")
        self.LPD_MODEL = YOLO("models/number_plate_detection.pt")
        self.LPR_MODEL = VisionEncoderDecoderModel.from_pretrained('models/number_plate_recognition/', local_files_only=True).to(self.DEVICE)
        self.PROCESSOR = TrOCRProcessor.from_pretrained('microsoft/trocr-small-printed')
        
        self.stopline = None # stopline uzunligi pixel
        self.stopline_direction = None # stopline ning zebradan tepada yoki pastda bo'lishi yo'nalishi
        self.stopline_meter = 4 # stop line uzunligi metr
        self.count_padestrian_for_violation = 1 # Qoidabuzarlik hisoblanishi uchun PO'Y da harakatlanayotgan piyodalar soni
        self.left_right_range_from_vehicle = 30 # Piyoda transport Boxidan shu masofa uzoqlikda bo'lsa qoidabuzarlik hisoblanmaydi
        self.right_left_distance_from_vehicle = 200 # Piyoda transport Boxiga shu masofadan yaqinroq bo'lsa qoidabuzarlik hisoblanadi
        self.camera_angle = 60
        self.two_meter_how_many_pixel = 460.
        
        self.zebra_polygon = Polygon([(250,640), (1920,620), (1920, 1080), (70,1080)])
        self.road_polygon = Polygon([(475,0), (1625,0), (1920,500), (1920,1080), (30,1080)])
        self.zebra_polygon_points = np.array([[250,640], [1920,620], [1920,1080], [70,1080]], np.int32).reshape((-1, 1, 2))
        self.road_polygon_points = np.array([[475,0], [1625,0], [1920,500], [1920,1080], [35,1080]], np.int32).reshape((-1, 1, 2))
        
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
        
            elif class_id==0:
                color = (0, 255, 255)

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
    
            cv2.rectangle(overlay, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (51,255,255), -1)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
            
        for vehicle in vehicles:
            index = list(track_ids).index(vehicle)
            xmin,ymin,xmax,ymax = boxes[index]
            
            cv2.rectangle(overlay, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0,0,204), -1)
            cv2.putText(frame, str(LP[vehicle]), (int(xmin), int(ymax)), cv2.FONT_HERSHEY_SIMPLEX , 1, (255,255,255), 2, cv2.LINE_AA)
            frame = cv2.addWeighted(overlay, 0.4, frame, 0.6, 0)
        
        return frame,vehicles

    def VISUALIZE(self, frame, boxes, track_ids, class_ids):
        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            color = (255,0,0)
            class_ = "Odam"
            
            if class_id in [2,5,7]: 
                color = (0, 0, 255)
                class_ = "Avtomobil"
                
            elif class_id==0: color = (0, 255, 255)
            
            cv2.rectangle(frame, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(frame, f"{class_} ({track_id})", (int(box[0]), int(box[1])-10), cv2.FONT_HERSHEY_SIMPLEX , 0.8, (255,255,255), 1, cv2.LINE_AA) 
            
            
    def run(self, source):
        cap = cv2.VideoCapture(source)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        track_history = defaultdict(lambda: [])
        LP = defaultdict(str)
        
        zebra_left_xy, zebra_right_xy, stopline = None, None, None
        out = cv2.VideoWriter('result.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 20, (width,height))
        temp = []
        while cap.isOpened():
            start = time.time()
            ret, frame = cap.read()

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
                cv2.line(frame, zebra_left_xy, zebra_right_xy, (0,0,255), 2)
                cv2.line(frame, (zebra_left_xy[0], zebra_left_xy[1]+self.stopline_direction*self.stopline), (zebra_right_xy[0], zebra_right_xy[1]+self.stopline_direction*self.stopline), (0,0,255), 2)
                
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
                    vehicles = []
                    if violator is not None and len(violator)>=1:
                        frame, vehicles = self.VISUALIZE_VIOLATOR(frame, boxes, track_ids, violator,LP)
                    
                    for vehicle in vehicles:
                        if LP[vehicle] not in temp and (len(LP[vehicle])==8 or len(LP[vehicle])==9):
                            temp.append(LP[vehicle])

                    if len(temp)>7:
                        temp.pop(0)

            except:
                pass

            coord = 30
            for lp in temp:
                cv2.putText(frame, lp, (30, coord), cv2.FONT_HERSHEY_SIMPLEX , 1, (0,0,255), 2, cv2.LINE_AA)
                coord+=30
            cv2.polylines(frame, [self.road_polygon_points], True, (255,255,255), 2)
            cv2.polylines(frame, [self.zebra_polygon_points], True, (255,255,255), 2)                
            
            end = time.time()

            out.write(frame)
            cv2.imshow("video", frame)
            ch = cv2.waitKey(1)
        
            if ch == 27 or ch == ord("q") or ch == ord("Q"):
                break
        cap.release()
        out.release()
        cv2.destroyAllWindows()

obj = PIYODA()
obj.run("DEMO.mp4")