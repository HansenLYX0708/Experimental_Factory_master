import os
import datetime
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, TerminateOnNaN

def callback_setting(logs_base_path, model_path, weight_only, if_log):
    time_stamp = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    log_dir = os.path.join(logs_base_path, time_stamp)
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    tbCallback = TensorBoard(log_dir=log_dir)
    save_checkpoint = ModelCheckpoint(filepath=model_path, save_best_only=True, save_weights_only=weight_only)
    early_stop = EarlyStopping(patience=30, min_delta=1e-6)
    stop_nan = TerminateOnNaN()

    if if_log:
        callback = [tbCallback, save_checkpoint, stop_nan, early_stop]
    else:
        callback = [save_checkpoint, stop_nan, early_stop]

    return callback