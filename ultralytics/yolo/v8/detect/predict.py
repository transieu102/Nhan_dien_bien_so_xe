# Ultralytics YOLO 🚀, GPL-3.0 license
import torch
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
lock=0
limit_len=13
text_glo=' '
def isTrueFormat(text):
     #form 5 so form 4 so
     if(len(text)==limit_len and text[2]=='-' and text[-4]=='.' and text[5]==' '\
        and '0'<=text[-2] and text[-2]<='9'\
            and '0'<=text[-3] and text[-3]<='9'\
                and '0'<=text[-5] and text[-5]<='9'\
                and '0'<=text[-6] and text[-6]<='9'\
                and '0'<=text[-7] and text[-7]<='9')\
        or (len(text)==11 and text[2]=='-' and text[5]==' '\
            and '0'<=text[-2] and text[-3]<='9'\
                and '0'<=text[-3] and text[-5]<='9'\
                and '0'<=text[-4] and text[-6]<='9'\
                and '0'<=text[-5] and text[-7]<='9'):
          return True
     return False
def merge_bboxes_nearby(bboxes, threshold_x=10, threshold_y=5):
    bboxes = [(i[0], i[1], i[2], i[3])for i in bboxes]
    bboxes = sorted(bboxes, key=lambda bbox: bbox[1])
    
    merged_bboxes = []
    current_bbox = None
    
    for bbox in bboxes:
        if current_bbox is None:
            # Initialize the current bounding box to the first bounding box in the list
            current_bbox = bbox
        elif abs(bbox[1] - current_bbox[1]) <= threshold_y :
            # The current bounding box and the next bounding box are nearby in both x and y direction
            # Merge the two bounding boxes by updating the x-coordinate of the right edge of the current bounding box
            current_bbox = (min(current_bbox[0],bbox[1]), min(current_bbox[1], bbox[1]), max(bbox[2],current_bbox[2]), max(current_bbox[3], bbox[3]))
        else:
            # The next bounding box is not nearby the current bounding box
            # Add the current bounding box to the list of merged bounding boxes and set the current bounding box to the next bounding box
            merged_bboxes.append(current_bbox)
            current_bbox = bbox
    
    # Add the last bounding box to the list of merged bounding boxes
    merged_bboxes.append(current_bbox)
    
    return merged_bboxes
class DetectionPredictor(BasePredictor):
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = (img if isinstance(img, torch.Tensor) else torch.from_numpy(img)).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_imgs):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det,
                                        classes=self.args.classes)

        results = []
        for i, pred in enumerate(preds):
            orig_img = orig_imgs[i] if isinstance(orig_imgs, list) else orig_imgs
            if not isinstance(orig_imgs, torch.Tensor):
                pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], orig_img.shape)
            path, _, _, _, _ = self.batch
            img_path = path[i] if isinstance(path, list) else path
            results.append(Results(orig_img=orig_img, path=img_path, names=self.model.names, boxes=pred))
        return results

    def write_results(self, idx, results, batch):
        global lock
        global text_glo
        global limit_len
        p, im, im0 = batch
        log_string = ''
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        imc = im0.copy() if self.args.save_crop else im0
        if self.source_type.webcam or self.source_type.from_img:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)
        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = results[idx].boxes  # TODO: make boxes inherit from tensors
        if len(det) == 0:
            lock=0
            text_glo=' '
            return f'{log_string}(no detections), '
        for c in det.cls.unique():
            n = (det.cls == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        # write

        for d in reversed(det):
            cls, conf, id = d.cls.squeeze(), d.conf.squeeze(), None if d.id is None else int(d.id.item())
            if self.args.save_txt:  # Write to file
                line = (cls, *d.xywhn.view(-1)) + (conf, ) * self.args.save_conf + (() if id is None else (id, ))
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')
            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                name = ('' if id is None else f'id:{id} ') + self.model.names[c]
                import requests
                text=''
                if lock==0:
                    image = save_one_box(d.xyxy,
                                imc,
                                file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                                BGR=True,save='false')
                    reg_url = "https://aiclub.uit.edu.vn/khoaluan/2022/khiemle/backend/recognizer/multipart"
                    #body = dict(file)
                    det_url = "https://aiclub.uit.edu.vn/gpu/service/paddleocr/predict_multipart"
                    import cv2
                    byte_img = cv2.imencode('.jpg', image)[1].tostring()
                    res = requests.post(
                                url=det_url, files=dict(binary_file=byte_img), data=dict(det=1,rec=0)
                            )
                    res = res.json()
                    list_bbox = res['predicts']
                    if not len(list_bbox):
                         continue
                    list_bbox = [list(map(int,i['bbox'])) for i in list_bbox]
                    list_bbox = merge_bboxes_nearby(list_bbox)
                    list_bbox = list(map(list,list_bbox))
                    # print(image)
                    for box in list_bbox:
                            box = list(map(int,box))
                            # w,h= image.shape[:2]
                            cropped = image[box[1]:box[3], box[0]:box[2]]
                            # cropped = image[box[1]:box[3], 0:w]
                            byte_cropped = cv2.imencode('.jpg', cropped)[1].tostring()
                            res = requests.post(
                                        url=reg_url, files=dict(file=byte_cropped)
                                    )
                            res = res.json()
                            if res['text']:
                                    text +=res['text']+' '
                    print(text)
                    print(len(text))
                    if isTrueFormat(text):
                                lock=1
                                text_glo=text
                    else:
                                lock=0
                label = None if self.args.hide_labels else (name if self.args.hide_conf else f'{name} {conf:.2f}')
                if len(text_glo)!=1:
                    self.annotator.box_label(d.xyxy.squeeze(), text_glo, color=colors(c, True))

                
  
            if self.args.save_crop:
                save_one_box(d.xyxy,
                             imc,
                             file=self.save_dir / 'crops' / self.model.model.names[c] / f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string


def predict(cfg=DEFAULT_CFG, use_python=False):
    model = cfg.model or 'yolov8n.pt'
    source = cfg.source if cfg.source is not None else ROOT / 'assets' if (ROOT / 'assets').exists() \
        else 'https://ultralytics.com/images/bus.jpg'

    args = dict(model=model, source=source)
    if use_python:
        from ultralytics import YOLO
        YOLO(model)(**args)
    else:
        predictor = DetectionPredictor(overrides=args)
        predictor.predict_cli()


if __name__ == '__main__':
    predict()
