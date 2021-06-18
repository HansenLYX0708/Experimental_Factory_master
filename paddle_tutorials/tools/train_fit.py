import paddle

from paddle_tutorials.modeling.models import MyModel
from paddle_tutorials.datasets.kaggle_facePoints import FaceDataset
from paddle_tutorials.datasets.MyDatasets import SliderSNDataset


img_path = 'C:/data/SliderSN/SliderSN_mnist/sn_x_train.npy'
label_path = 'C:/data/SliderSN/SliderSN_mnist/sn_y_train.npy'

train_dataset = SliderSNDataset(img_path, label_path, mode='train')
val_dataset = SliderSNDataset(img_path, label_path, mode='val')


#train_loader = paddle.io.DataLoader(train_dataset, batch_size=256)
#val_loader = paddle.io.DataLoader(val_dataset, batch_size=256)


model = paddle.Model(MyModel())

model.summary(input_size=[1,1,40,24])

optim = paddle.optimizer.Adam(learning_rate=1e-3,
    parameters=model.parameters())
loss = paddle.nn.CrossEntropyLoss(use_softmax=True)
model.prepare(optim, loss)
model.fit(train_dataset, val_dataset,  batch_size=256, epochs=60, verbose=1)

