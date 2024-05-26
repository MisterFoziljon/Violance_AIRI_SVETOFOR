import argparse
from shapely.geometry.polygon import Polygon
import os

def parse_args():
    PATH = os.getcwd()
    parser = argparse.ArgumentParser(description="PIYODALARGA NISBATAN YO'L QOIDASI BUZILISHINI ANIQLOVCHI DASTUR PARAMETRLARI")
    
    parser.add_argument(
        "--device_name", 
        type=str, 
        default='cuda:0', 
        help="The path of the facial detection model")
    
    parser.add_argument(
        "--main_model",
        type=str, 
        default=os.path.join('models/yolov8x.pt'),
        help="The path of the facial recognition model")

    parser.add_argument(
        "--zebra_model",
        type=str, 
        default=os.path.join('models/zebra.pt'),
        help="The path of the facial recognition model")

    parser.add_argument(
        "--lpd_model",
        type=str, 
        default=os.path.join('models/number_plate_detection.pt'),
        help="The path of the facial recognition model")

    parser.add_argument(
        "--lpr_model",
        type=str, 
        default='models/number_plate_recognition/',
        help="The path of the facial recognition model")

    parser.add_argument(
        "--processor",
        type=str, 
        default=os.path.join('microsoft/trocr-small-printed'),
        help="The path of the facial recognition model")
    
    parser.add_argument(
        "--stopline_meter",
        type=int,
        default=0,
        help="video source path or rtsp protocol")

    parser.add_argument(
        "--count_padestrian_for_violation",
        type=int,
        default=1,
        help="video source path or rtsp protocol")
    
    parser.add_argument(
        "--left_right_range_from_vehicle",
        type=float, 
        default=50)

    parser.add_argument(
        "--right_left_distance_from_vehicle",
        type=float, 
        default=200)

    parser.add_argument(
        "--camera_angle",
        type=float, 
        default=60)

    parser.add_argument(
        "--two_meter_how_many_pixel",
        type=float, 
        default=460)
    
    parser.add_argument(
        "--zebra_polygon", 
        type=Polygon,
        default=Polygon([(250,640), (1920,620), (1920, 1080), (70,1080)]))

    parser.add_argument(
        "--zebra_polygon_points", 
        type=list,
        default=[[250,640], [1920,620], [1920,1080], [70,1080]])

    parser.add_argument(
        "--road_polygon", 
        type=Polygon,
        default=Polygon([(475,0), (1625,0), (1920,500), (1920,1080), (30,1080)]))
    
    parser.add_argument(
        "--road_polygon_points", 
        type=list,
        default=[[475,0], [1625,0], [1920,500], [1920,1080], [35,1080]])
    
    args = parser.parse_args()

    return args

def main():
    FLAGS = parse_args()

if __name__ == '__main__':
    main()