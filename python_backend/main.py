import os
import sys
import numpy as np
import json
import torch
import cv2
from ultralytics import YOLO
import warnings

os.chdir('/python_backend')

from classes import Detect_Image
from classes import Convert_Excel
from classes import Final_Sheet
from classes import Delete_Files

def detectarImagem(arg1, config, arg2, arg3):
    detection = Detect_Image(arg1, config, arg2, arg3)
    
    detection.area, detection.height_source, detection.width_source = detection.calcPdfSize()

    detection.dpi = detection.calcDpi()

    detection.pdf_image_path = detection.calcParameters()

    detection.image_src, detection.height_img, detection.width_img, detection.image_name = detection.processarImagem()

    detection.lines_horizontal, detection.minimum, detection.constante, detection.tolerancia_ = detection.detect_lines_and_save()

    detection.all_rect = detection.calcAll_rect(
        detection.lines_horizontal, 
        detection.tolerancia_, 
        detection.height_img, 
        detection.minimum
        )
    
    detection.new_rect = detection.all_rect[((abs(detection.all_rect[:,2] - detection.all_rect[:,0])) > (detection.width_img * detection.config['minShape'])) & ((abs(detection.all_rect[:,3] - detection.all_rect[:,1])) > (detection.height_img * detection.config['minShape']))]
    detection.new_rect = detection.new_rect[((abs(detection.new_rect[:,2] - detection.new_rect[:,0])) < (detection.width_img * detection.config['maxShape'])) | ((abs(detection.new_rect[:,3] - detection.new_rect[:,1])) < (detection.height_img * detection.config['maxShape']))]

    detection.non_overlapping_rectangles = detection.processamento_retangulos(
        detection.new_rect,
        detection.config['intersectionArea']
        )

    detection.non_overlapping_rectangles = np.unique(detection.non_overlapping_rectangles, axis=0)

    detection.save_detections()

    return detection

def yoloDetect(filename_id):
    images = []
    for arquivo in os.listdir('./cropped_images'):
        if filename_id in arquivo:
            caminho_relativo = os.path.join('./cropped_images', arquivo)
            caminho_absoluto = os.path.abspath(caminho_relativo)
            images.append(caminho_absoluto)

    results = model(images, conf=0.7)
    j = 0
    
    for idx, result in enumerate(results):
        image = cv2.imread(images[idx])
        boxes = result.boxes
        for box_idx, box in enumerate(boxes):
            xyxy = box.xyxy[0]
            x1, y1, x2, y2 = xyxy[0].item(), xyxy[1].item(), xyxy[2].item(), xyxy[3].item()
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            crop_img = image[y1:y2, x1:x2]
            cv2.imwrite(f'./crops2/{j}{filename_id}.png', crop_img)
            j += 1

def excel(current_client, filename_id):
    tables = Convert_Excel(current_client, filename_id)
    tables.excel()

    final_sheet = Final_Sheet(current_client, filename_id)

    warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl.styles.stylesheet")
    if current_client == 'Caterpillar':
        final_sheet.cat_convert()
    elif current_client == 'Whirlpool':
        final_sheet.whirlpool_convert()
    else:
        final_sheet.generic_convert()

def deleteFiles(path, filename_id, final_sheet=False):
    delet = Delete_Files(final_sheet, path, filename_id)
    delet.deleteFiles()

if __name__ == '__main__':
    arg1 = sys.argv[1]
    arg2 = sys.argv[2]
    with open('./config.json') as json_data:
        data = json.load(json_data,)
        config = data['customers'][arg2]
    arg3 = sys.argv[3]

    detection =  detectarImagem(arg1, config, arg2, arg3)
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = YOLO('./Models/newS.pt').to(device)

    if detection.current_client != 'HPE' and detection.current_client != 'Whirlpool':
        yoloDetect(detection.filename_id)
    
    excel(detection.current_client, detection.filename_id)

    print('7', flush = True)

    deleteFiles('./cropped_images', detection.filename_id)
    deleteFiles('./crops2', detection.filename_id)
    deleteFiles('./Excel', detection.filename_id, True)
    deleteFiles('./processed_images', detection.filename_id)
    deleteFiles('./images', detection.filename_id)

    print('8', flush = True)

