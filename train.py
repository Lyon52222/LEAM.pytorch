import config
from data import dataloader
from data.dataloader import LEAMDataLoader
from models.LEAM import *
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm

def calculate_acc(y_pred, targets):
    _, y_pred = torch.max(y_pred, -1)
    n_correct = torch.eq(y_pred, targets).sum().item()
    return n_correct / len(y_pred) * 100


def make_train_state(args):
    return {
        'epoch_index': 0,
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'test_loss': -1,
        'test_acc': -1
    }

def train(opt):
    device = torch.device(opt.device)
    loader = LEAMDataLoader(opt)

    train_iter, val_iter, test_iter = loader.get_loader()
    
    model = LEAM(loader.get_text_vocab(), loader.get_label_text_vocab(), opt.embedding_size, opt.output_size, opt.hidden_size, opt.r)

    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=opt.learning_rate)
    loss_func = nn.CrossEntropyLoss()

    train_state = make_train_state(opt)

    epoch_bar = tqdm.tqdm(desc='training routine', total=opt.num_epochs)
    train_bar = tqdm.tqdm(desc='train', total=len(train_iter), position=1, leave=True)
    val_bar = tqdm.tqdm(desc='val', total=len(val_iter), position=1, leave=True)

    running_loss = 0.0
    running_acc = 0.0
    model.train()

    try:
        for epoch_index in range(opt.num_epochs):
            train_state['epoch_index'] = epoch_index

            for batch_index, batch_dict in enumerate(train_iter):
                optimizer.zero_grad()

                #y_pred = model(x=batch_dict.text)
                y_pred = model(batch_dict.text, batch_dict.label_text)
                #print(y_pred) 
                #print(batch_dict.label)

                loss = loss_func(y_pred, batch_dict.label.long())
                loss_t = loss.item()

                running_loss += (loss_t - running_loss) / (batch_index + 1)

                loss.backward()

                optimizer.step()

                acc_t = calculate_acc(y_pred, batch_dict.label.float())
                running_acc += (acc_t - running_acc) / (batch_index + 1)

                train_bar.set_postfix(epoch = epoch_index, loss = running_loss, acc = running_acc)
                train_bar.update()

            train_state['train_loss'].append(running_loss)
            #train_state['train_acc'].append(running_acc)


            running_acc = 0.0
            running_loss = 0.0
            model.eval()
            
            for batch_index, batch_dict in enumerate(val_iter):
                '''
                if batch_index > 3:
                    break
                '''
                #y_pred = model(batch_dict.text)
                y_pred = model(batch_dict.text, batch_dict.label_text)

                loss = loss_func(y_pred, batch_dict.label.long())
                loss_t = loss.item()

                running_loss += (loss_t-running_loss) / (batch_index + 1)

                acc_t = calculate_acc(y_pred, batch_dict.label.float())
                running_acc += (acc_t - running_acc) / (batch_index + 1)

                val_bar.set_postfix(epoch = epoch_index, loss=running_loss, acc=running_acc)
                val_bar.update()

            train_state['val_loss'].append(running_loss)

            train_bar.n = 0
            val_bar.n = 0

            epoch_bar.update()

    except KeyboardInterrupt:
        print("Exiting loop")

if __name__ == '__main__':
    opt = config.parse_opt()
    train(opt)