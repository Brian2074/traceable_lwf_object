import torch.nn as nn
import models.layers as nl
import torch.nn.functional as F
import torch
import gc

from models.ServerDataset import ServerDataset
from torch.utils.data import DataLoader
from models.ServerModel import KL_Loss
from utils import Optimizers, set_logger, Metric, classification_accuracy, remove_logger
from tqdm import tqdm

def train_and_get_message(server_model, messages_to_server, args):
    messages_to_clients = []
    task_id = messages_to_server[0]['task_id']
    server_model.set_dataset(task_id)

    extracted_feature_list = []
    logits_list = []
    label_list = []
    extracted_feature_list_test = []
    labels_list_test = []

    for msg in messages_to_server:
        extracted_feature_list.extend(msg['extracted_feature_list'])
        logits_list.extend(msg['logits_list'])
        label_list.extend(msg['label_list'])
        extracted_feature_list_test.extend(msg['extracted_feature_list_test'])
        labels_list_test.extend(msg['labels_list_test'])
    
    # Validate client data
    print(f"Received {len(extracted_feature_list)} training samples from clients")
    if len(extracted_feature_list) > 0:
        sample_feature = torch.stack(extracted_feature_list[:min(10, len(extracted_feature_list))])
        sample_logits = torch.stack(logits_list[:min(10, len(logits_list))])
        if not torch.isfinite(sample_feature).all():
            print(f"Warning: Client features contain NaN/Inf!")
        if not torch.isfinite(sample_logits).all():
            print(f"Warning: Client logits contain NaN/Inf!")

    train_dataloader = get_train_dataloader(extracted_feature_list, logits_list, label_list, args.batch_size, True)
    test_dataloader = get_test_dataloader(extracted_feature_list_test, labels_list_test, args.batch_size)

    opt = torch.optim.SGD(server_model.parameters(), lr=0.001, weight_decay=0.00001)

    for epoch_idx in range(args.epochs_server):
        train_server_model(server_model, train_dataloader, epoch_idx, task_id, opt, args)
        test_server_model(server_model, test_dataloader, epoch_idx, task_id, args)
    
    # Explicitly delete large objects to free memory
    del train_dataloader, test_dataloader
    del extracted_feature_list, logits_list, label_list
    del extracted_feature_list_test, labels_list_test
    
    # Clear GPU cache after training
    if args.cuda:
        torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()
    
    for msg in messages_to_server:
        msg_to_client = {
            'client_id': msg['client_id'],
            'task_id': msg['task_id']
        }
        
        msg_dataloader = get_train_dataloader(msg['extracted_feature_list'], msg['logits_list'], msg['logits_list'], args.batch_size, False)
        msg_to_client['output_logits'] = get_outputs_logits(server_model, msg_dataloader, msg['client_id'], msg['task_id'], args)
        
        # Delete dataloader immediately after use
        del msg_dataloader
        
        messages_to_clients.append(msg_to_client)

    # Clear GPU cache before returning
    if args.cuda:
        torch.cuda.empty_cache()
    
    # Force garbage collection
    gc.collect()

    return messages_to_clients

def train_server_model(server_model, train_dataloader, epoch_idx, task_id, opt, args):
    train_accuracy = Metric('train_accuracy')

    if args.cuda:
        server_model.cuda(args.device)
    server_model.train()
    
    # Initialize GradScaler for mixed precision
    scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)

    with tqdm(total=len(train_dataloader),
                desc='Server Train Ep. #{}: '.format(epoch_idx + 1),
                disable=False,
                ascii=True) as t:
        for batch_idx, (indexs, data, target, logits) in enumerate(train_dataloader):
            opt.zero_grad()
            if args.cuda:
                data, logits, target = data.cuda(args.device), logits.cuda(args.device), target.cuda(args.device)

            num = data.size(0)
            
            # Use automatic mixed precision for forward pass
            with torch.amp.autocast('cuda', enabled=args.use_amp):
                output = server_model(data)

                # loss
                loss_true = server_model.compute_loss(output, target)
                loss_KL = server_model.criterion_KL(output, logits)
                
                # Check for nan/inf and clip if needed
                if not torch.isfinite(loss_true):
                    print(f"Warning: loss_true is {loss_true}, skipping batch")
                    continue
                if not torch.isfinite(loss_KL):
                    print(f"Warning: loss_KL is {loss_KL}, skipping batch")
                    continue

                loss = loss_true * args.a_s + loss_KL * (1 - args.a_s)
            
            # Scale loss and backward pass
            scaler.scale(loss).backward()

            label = target.max(1, keepdim=True)[1]
            train_accuracy.update(classification_accuracy(output, label), num)
            
            # Clip gradients to prevent explosion
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(server_model.parameters(), max_norm=1.0)
            
            # Unscale gradients and step optimizer
            scaler.step(opt)
            scaler.update()
            t.set_postfix({'loss': loss.item(),
                            'accuracy': '{:.2f}'.format(100. * train_accuracy.avg.item())})
            t.update(1)

def test_server_model(server_model, test_dataloader, epoch_idx, task_id, args):
    test_accuracy = Metric('test_accuracy')

    if args.cuda:
        server_model.cuda(args.device)
    server_model.eval()

    with tqdm(total=len(test_dataloader),
            desc='Server Eval Ep. #{}: '.format(epoch_idx + 1),
            disable=False,
            ascii=True) as t:
        for batch_idx, (indexs, data, target) in enumerate(test_dataloader):
            if args.cuda:
                data, target = data.cuda(args.device), target.cuda(args.device)
            num = data.size(0)
            output = server_model(data)

            label = target.max(1, keepdim=True)[1]
            test_accuracy.update(classification_accuracy(output, label), num)
            t.set_postfix({
                        'accuracy': '{:.2f}'.format(100. * test_accuracy.avg.item())})
            t.update(1)

def get_outputs_logits(server_model, msg_dataloader, client_id, task_id, args):
    if args.cuda:
        server_model.cuda(args.device)
    server_model.eval()
    buffer = []

    with tqdm(total=len(msg_dataloader),
            desc='Get Output Logits. #{}: '.format(client_id),
            disable=False,
            ascii=True) as t:
        for batch_idx, (index, data, target, logits) in enumerate(msg_dataloader):
            if args.cuda:
                data = data.cuda(args.device)
            num = data.size(0)
            output = server_model(data)
            buffer += output.cpu()
            t.update(1)
            
    return buffer

def get_train_dataloader(extracted_feature_list, logits_list, label_list, batchsize, shuffle):
    train_dataset = ServerDataset(extracted_feature_list, label_list, logits_list, [], [])
    train_loader = DataLoader(dataset=train_dataset,
                                  shuffle=shuffle,
                                  batch_size=batchsize,
                                  pin_memory=True)
    return train_loader

def get_test_dataloader(extracted_feature_list_test, labels_list_test, batchsize):
    test_dataset = ServerDataset([], [], [], extracted_feature_list_test, labels_list_test)
    test_loader = DataLoader(dataset=test_dataset,
                                shuffle=False,
                                batch_size=batchsize,
                                pin_memory=True)
    return test_loader