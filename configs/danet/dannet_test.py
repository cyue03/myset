_base_ = [
    '../_base_/models/danet_r50-d8.py', '../_base_/datasets/aeroscape.py',
    '../_base_/default_runtime.py', '../_base_/schedules/schedule_20k.py'
]
crop_size = (1280, 720)
data_preprocessor = dict(size=crop_size)
model = dict(
	 data_preprocessor=data_preprocessor,decode_head=dict(num_classes=11),auxiliary_head=dict(num_classes=11))