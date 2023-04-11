from mmseg.registry import DATASETS
from .basesegdataset import BaseSegDataset
import os.path as osp

@DATASETS.register_module()
class MyDataset(BaseSegDataset):
  CLASSES = ("person","bike","Car","Drone","Boat","Animal","Obstacle","Construction","Vegetation","Road","Sky"),
  PALETTE = [[192,128,128],[0,128,0],[128,128,128],[128,0,0],[0,0,128],[192,0,128],[192,0,0],[192,128,0],[0,64,0],[128,128,0],[0,128,128]]
  def __init__(self, **kwargs):
    super().__init__(img_suffix='.jpg', seg_map_suffix='.png', data_prefix=dict(img_path='JPEGImages/',seg_map_path='SegmentationClass/'),
                      **kwargs)

    assert osp.exists(self.data_prefix['img_path'])