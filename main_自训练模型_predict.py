from detect import parse_opt, main
import pandas as pd


def toPandas(det):
    ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class'  # xyxy columns
    return pd.DataFrame(det, columns=ca)


opt = parse_opt()
opt.weights = "/Users/zhangxuewei/Documents/GitHub/yolov5/yolov5s_backup.pt"  # 模型参数
opt.source = "/Users/zhangxuewei/Documents/GitHub/yolov5/data/image_backup"   # 识别对象
opt.data = "/Users/zhangxuewei/Documents/GitHub/yolov5/data/coco128.yaml"     # 类别名
opt.save_txt = False
opt.view_img = False
det = main(opt)

print(toPandas(det))
