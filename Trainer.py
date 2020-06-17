from args import args
import torch
import torch.nn as nn
from loadData import DataSet
import os
from model import train, test, pretrain_model, init_models
from utils.misc import save_checkpoint

# device set up
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('device = ', device)
# get args
params = {k: v for k,v in args._get_args()}
print('params = ', params)

# DataSet set up
num = 2000
img_sz = 28 if (params['category'] == 'Fashion-Mnist' or params['category'] == 'Mnist') else 64
batch_sz = 128
data = DataSet(num=num, img_size=img_sz, batch_size=batch_sz, category=params['category'])

def run(model, data):

    # set up the loss function
    criterion = nn.MSELoss()

    # set up the optimizer
    optim = torch.optim.Adam(model.parameters(), lr=params['lr'])

    # epoch to start
    start = 1
    if params['resume']:
        print('Loading from latest checkpoint ...')
        assert os.path.isfile(params['latest_ckpt']), 'Error: no checkpoint directory found!!'
        ckpt = torch.load(params['latest_ckpt'])

        """
        
        others to reload
        
        """

        # params that must to be reloaded
        start = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        optim.load_state_dict((ckpt['optimizer']))

    if params['test']:
        print('\n Test only')
        _ = test(data, model, criterion)
        """
        
        visualize...
        
        """
        return

    for epoch in range(start, params['epochs']):
        print('[{}/{}]'.format(epoch, params['epochs']))

        _ = train(data, model, criterion, optim)

        """
        
        visualize,
        evaluate,
        ...
        
        """

        print("loss: %f, ..." % ( ... ))
        save_checkpoint({
            """
            
            others to load
            
            """
            
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optim.state_dict()
            },
            checkpoint=params['ckpt_dir'])


if __name__ == '__main__':
    if params['pretrain']:
        model = pretrain_model(model_name=params['pretrain_model'], feature_extract=True)
    else:
        model = init_models(params)

    model.to(device)
    print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters()) / 1000000.0))

    run(model, data=data)











