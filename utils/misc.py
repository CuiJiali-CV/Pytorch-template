import torch
import os

def save_checkpoint(state, is_best, checkpoint='checkpoint', filename='checkpoint.pth.tar'):

    if not os.path.exists(checkpoint):
        os.makedirs(checkpoint)
    # 保存断点信息
    filepath = os.path.join(checkpoint, filename)
    print('checkpoint filepath = ',filepath)
    torch.save(state, filepath)
    # 模型保存
    if is_best:
        model_name = 'garbage_resnext101_model_' + str(state['epoch']) + '_' + str(
            int(round(state['train_acc'] * 100, 0))) + '_' + str(
            int(round(state['test_acc'] * 100, 0))) + '.pth'
        print('Validation loss decreased  Saving model ..,model_name = ', model_name)
        model_path = os.path.join(checkpoint, model_name)
        print('model_path = ',model_path)
        torch.save(state['state_dict'], model_path)