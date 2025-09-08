# Example of modifying a config to use your custom dataset:
# Assuming you are using a file like `rtmdet_tiny_1xb32-300e_coco.py`

# Change the dataset type to CocoDataset
train_dataloader = dict(
    dataset=dict(
        type='CocoDataset',
        ann_file='TrainingData/annotations_coco.json',  # Path to your custom COCO annotations
        data_root='TrainingData/',  # Path to your images directory
        img_prefix='images/',
        metainfo=dict(classes=('card',)),  # Class names for your cards
    )
)

# Define the number of classes
model = dict(
    bbox_head=dict(
        num_classes=1  # 1 class for "card"
    )
)
