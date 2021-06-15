import paddle

from paddle_tutorials.modeling.models import MyModel
from paddle_tutorials.datasets.kaggle_facePoints import FaceDataset

model = paddle.Model(MyModel())
model.summary((1,1, 96, 96))


Train_Dir = 'C:/data/facial-keypoints-detection/training.csv'
Test_Dir = 'C:/data/facial-keypoints-detection/test.csv'
lookid_dir = 'C:/data/facial-keypoints-detection/IdLookupTable.csv'


# 训练数据集和验证数据集
train_dataset = FaceDataset(Train_Dir, mode='train')
val_dataset = FaceDataset(Train_Dir, mode='val')

# 测试数据集
test_dataset = FaceDataset(Test_Dir, mode='test')


model = paddle.Model(MyModel())

optim = paddle.optimizer.Adam(learning_rate=1e-3,
    parameters=model.parameters())
model.prepare(optim, paddle.nn.MSELoss())
model.fit(train_dataset, val_dataset, eval_freq=10, epochs=60, batch_size=256, verbose=1)

