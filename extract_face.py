from face_det_torch.predict_model import FaceDet
import cv2
import numpy as np
import PIL
import glob
from pathlib import Path
import os
import shutil
import threading



def get_output_path(path):
    parts = path.split('/')
    parts[0] = 'data_extracted'
    out_path = '/'.join(parts)
    return out_path



def clean_output_dir():
    path = Path('data_extracted/')
    if os.path.exists(path):
        shutil.rmtree(str(path))
    path.mkdir(parents=True)

    os.mkdir('data_extracted/0')
    os.mkdir('data_extracted/1')


face_det = FaceDet()

def extract_face_in_file(in_path, out_path):
    img_raw = cv2.imread(in_path, cv2.IMREAD_COLOR)
    img = np.float32(img_raw)
    predicts = face_det.predict(img, visual_mode=False)
    
    if len(predicts) == 0:
        return
    
    out_img = predicts[0][0]
    
    try:
        cv2.imwrite(out_path, out_img)
    except Exception as e:
        print('write failed: ', e, in_path, out_path)



def extract_face_in_dir(path):
    for path in Path(path).rglob('*.jpg'):
        path = str(path)
        out_path = get_output_path(path)
        extract_face_in_file(path, out_path)



def main():
    clean_output_dir()

    t0 = threading.Thread(target=extract_face_in_dir, args=('data/0',))
    t1 = threading.Thread(target=extract_face_in_dir, args=('data/1',))
    t0.start()
    t1.start()

    t0.join()
    t1.join()



if __name__ == "__main__":
    main()