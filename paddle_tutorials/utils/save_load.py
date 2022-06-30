import os
import paddle
from paddle.static import load_program_state

def _mkdir_if_not_exist(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except OSError as e:
            raise OSError('Failed to mkdir {}'.format(path))

def _save_student_model(net, model_prefix):
    student_model_prefix = model_prefix + "_student.pdparams"
    if hasattr(net, "_layers"):
        net = net._layers
    if hasattr(net, "student"):
        paddle.save(net.student.state_dict(), student_model_prefix)

def save_model(net, opt, model_path, epoch_id, prefix ='classification'):
    if paddle.distributed.get_rank() != 0:
        return

    _mkdir_if_not_exist(model_path)
    model_path = os.path.join(model_path, str(epoch_id))

    model_prefix = os.path.join(model_path, prefix)
    _save_student_model(net, model_prefix)

    paddle.save(net.state_dict(), model_path + ".pdparams")
    paddle.save(opt.state_dict(), model_path + ".pdopt")

def load_dygraph_pretrain(model, path, load_static_weights):
    if not (os.path.isdir(path) or os.path.exists(path + '.pdparams')):
        return
    if load_static_weights:
        pre_state_dict = load_program_state(path)
        params_state_dict = {}
        model_dict = model.state_dict()
        for key in model_dict.keys():
            weight_name = model_dict[key].name
            if weight_name in pre_state_dict.keys():
                params_state_dict[key] = pre_state_dict[weight_name]
            else:
                params_state_dict[key] = model_dict[key]
        return
    params_state_dict = paddle.load(path + ".pdparams")
    model.set_dict(params_state_dict)
    return

def init_model(net, checkpoints=None, opt=None, pretrainmodel=None, load_static_weights=False, use_distillation=False):
    if checkpoints and opt is not None:
        para_dict = paddle.load(checkpoints + ".pdparams")
        opti_dict = paddle.load(checkpoints + ".pdopt")
        net.set_dict(para_dict)
        opt.set_state_dict(opti_dict)
        return
    if pretrainmodel:
        if use_distillation:
            # TODO add functions
            return
        else:
            load_dygraph_pretrain(net, path=pretrainmodel, load_static_weights=load_static_weights)


