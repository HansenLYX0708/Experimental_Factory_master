# Base code
# Add save model
# Add visual DL
import numpy as np
import os
import paddle
import paddle.nn.functional as F
from paddle.metric import accuracy
from visualdl import LogWriter

from paddle_tutorials.modeling.models import MyModel
from paddle_tutorials.datasets.MyDatasets import SliderSNDataset
from paddle_tutorials.utils.save_load import save_model, init_model

#train_loader = paddle.io.DataLoader(train_dataset, batch_size=256)
#val_loader = paddle.io.DataLoader(val_dataset, batch_size=256)


if __name__ == '__main__':
    # config
    img_path = 'C:/data/SliderSN/SliderSN_mnist/sn_x_train.npy'
    label_path = 'C:/data/SliderSN/SliderSN_mnist/sn_y_train.npy'
    epochs = 10
    model_save_path = 'C:/model_outputs/cnn_sn/model_save'
    vdl_writer_path = 'C:/model_outputs/cnn_sn/vdl_log'
    val_freq = 3

    is_load_model = False
    checkpoints = "C:/model_outputs/cnn_sn/model_save/9"

    is_save_model = True
    batch_size = 256





    # Get dataset
    train_dataset = SliderSNDataset(img_path, label_path, mode='train')
    val_dataset = SliderSNDataset(img_path, label_path, mode='val')

    train_loader = paddle.io.DataLoader(train_dataset, batch_size=batch_size)
    val_loader = paddle.io.DataLoader(val_dataset, batch_size=batch_size)

    model = MyModel()

    test_model = paddle.Model(model)
    test_model.summary((1, 1, 40, 24))

    optim = paddle.optimizer.Adam(learning_rate=1e-3, parameters=model.parameters())
    vdl_writer = LogWriter(vdl_writer_path)


    # load pretrained model
    if is_load_model:
        init_model(model, checkpoints=checkpoints, opt=optim)


    # trainning
    total_step = 0
    val_total_step = 0
    best_loss = np.Inf
    for epoch in range(1, epochs):
        for batch_id, data in enumerate(train_loader):
            x_data, y_data = data
            predicts = model(x_data)
            loss = F.cross_entropy(predicts, y_data, use_softmax=False)
            acc = accuracy(predicts, y_data)
            loss.backward()



            if batch_id % 50 == 0:
                img_input = x_data.numpy()[0][0]
                img_input = np.expand_dims(img_input, 2)
                conv_2 = model.conv2_output(x_data).numpy()[0][0]
                conv_2 = np.expand_dims(conv_2, 2)
                conv_3 = model.conv3_output(x_data).numpy()[0][0]
                conv_3 = np.expand_dims(conv_3, 2)

                #global total_step
                vdl_writer.add_scalar(tag="train_acc", step=total_step, value=acc.numpy()[0])
                vdl_writer.add_scalar(tag="train_loss", step=total_step, value=loss.numpy()[0])
                #vdl_writer.add_image(tag='input_img', img=img_input, step=total_step)
                #vdl_writer.add_image(tag='conv2', img=conv_2, step=total_step)
                #vdl_writer.add_image(tag='conv3', img=conv_3, step=total_step)
                total_step += 1
                print("epoch : {}, batch_id : {}, acc : {}, loss : {}".format(epoch, batch_id, acc.numpy(), loss.numpy()))
            optim.step()
            optim.clear_grad()

        if epoch % val_freq == 0:
            for val_batch_id, val_data in enumerate(val_loader):
                val_x_data, val_y_data = val_data
                val_predicts = model(val_x_data)
                val_loss = F.cross_entropy(val_predicts, val_y_data, use_softmax=False)
                val_acc = accuracy(val_predicts, val_y_data)
                if val_batch_id % 50 == 0:
                    vdl_writer.add_scalar(tag="val_acc", step=val_total_step, value=val_acc)
                    vdl_writer.add_scalar(tag="val_loss", step=val_total_step, value=val_loss)
                    val_total_step += 1
                    print("val epoch : {}, acc : {}, loss : {}".format(epoch, val_acc.numpy(), val_loss.numpy()))

        if is_save_model and (loss.numpy() < best_loss):
            best_loss = loss.numpy()
            save_model(model, optim, model_save_path, epoch, prefix='')

    vdl_writer.close() if vdl_writer else None




