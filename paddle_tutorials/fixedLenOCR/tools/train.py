import pandas as pd
import paddle
from paddle.metric import Accuracy
from paddle.vision.models import resnet18
import paddle.vision.transforms as T
import warnings


from paddle_tutorials.fixedLenOCR.dataset.MyDataset import MyDataset


warnings.filterwarnings("ignore")


df = pd.read_csv('C:\\data\\foods\\food_data.csv')
image_path_list = df['images'].values
label_list = df['label'].values

all_size = len(image_path_list)
train_size = int(all_size * 0.8)
train_image_path_list = image_path_list[:train_size]
train_label_list = label_list[:train_size]
val_image_path_list = image_path_list[train_size:]
val_label_list = label_list[train_size:]

transform = T.Compose([
    T.Resize([224, 224]),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # T.Transpose(),
])

train_dataset = MyDataset(image=train_image_path_list, label=train_label_list, transform=transform)

train_loader = paddle.io.DataLoader(train_dataset, places=paddle.CPUPlace(), batch_size=8, shuffle=True)

val_dataset = MyDataset(image=val_image_path_list, label=val_label_list, transform=transform)

val_loader = paddle.io.DataLoader(val_dataset, places=paddle.CPUPlace(), batch_size=8, shuffle=True)

# build model
model = resnet18(pretrained=True, num_classes=5, with_pool=True)

model = paddle.Model(model)
optim = paddle.optimizer.Adam(learning_rate=0.001, parameters=model.parameters())

model.prepare(
    optim,
    paddle.nn.CrossEntropyLoss(),
    Accuracy(topk=(1, 2))
    )

model.fit(train_loader, val_dataset, epochs=2, verbose=1)

model.evaluate(train_dataset, batch_size=8, verbose=1)


model.save('inference_model', False)