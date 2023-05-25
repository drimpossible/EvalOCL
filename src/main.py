import os, torch, random, warnings
import numpy as np
from os.path import exists
from opts import parse_args
from torch.optim import SGD
import torchvision.models as models
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision import transforms
from torch.utils.data import DataLoader
from components import ListFolder, EmbedLinearClassifier, LinearLR
from utils import get_logger, seed_everything, save_model
import torch.nn as nn
import copy
warnings.filterwarnings('ignore')

# Global variables 
test_transforms = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
train_transforms = transforms.Compose([transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

# Compute and memory optimized version of the codebase: Real-Time Evaluation in Online Continual Learning: A New Hope
def ER_train(opt, model, logger):
    # Load pretraining data
    os.makedirs(opt.log_dir+'/'+opt.exp_name+'/', exist_ok=True)
    assert(opt.num_per_chunk%opt.test_batch_size==0)

    if exists(opt.log_dir+'/'+opt.exp_name+f'/labels.npy'):
        labelarr, predarr, idxarr = np.load(opt.log_dir+'/'+opt.exp_name+'/labels.npy'), np.load(opt.log_dir+'/'+opt.exp_name+'/preds.npy'), np.load(opt.log_dir+'/'+opt.exp_name+'/curr_idx.npy')
        model.load_state_dict(torch.load(opt.log_dir+'/'+opt.exp_name+f'/model.pt')['state_dict'])
    else:
        labelarr, predarr = np.zeros(len(test_dataset.labels),dtype='u2'), np.zeros(len(test_dataset.labels),dtype='u2')
        idxarr = np.zeros(2,dtype='int') # train and test shift respectively, for resuming experiments
        idxarr[1]+= opt.delay * opt.test_batch_size # additional delay for shifted evaluation   

    trainoffset, testoffset = idxarr[0], idxarr[1]    
    max_idx = testoffset//opt.num_per_chunk
    logger.info('==> Training starting from idx '+str(trainoffset)+' and '+str(testoffset)+'..')

    # TODO: feature extraction is not fully tested yet
    if opt.extract_feats: featarr = np.zeros((opt.num_per_chunk, opt.embed_size),dtype=np.float32)

    # For sensitivity analysis experiments only, comment out for most experiments
    #image_paths = opt.order_file_dir+'/pretrain_image_paths.hdf5'
    #y = opt.order_file_dir+'/pretrain_labels.hdf5'

    image_paths = opt.order_file_dir+'/train_image_paths.hdf5'
    y = opt.order_file_dir+'/train_labels.hdf5'
    
    train_routing_index_path = f'{opt.order_file_dir}/{opt.train_batch_size}_{opt.test_batch_size}_{opt.sampler}_{opt.num_gdsteps}.hdf5'
    
    isACE=False
    if opt.fc=="ACE": 
        isACE = True
        train_mask = f'{opt.order_file_dir}/mask_{opt.train_batch_size}_{opt.test_batch_size}_{opt.sampler}_{opt.num_gdsteps}.hdf5'

    # Load dataloaders
    train_dataset = ListFolder(root=opt.data_dir, image_paths=image_paths, labels_path=y, routing_idx_path=train_routing_index_path, offset=trainoffset, is_train=True, transform=train_transforms, isACE=isACE, mask_values_path=train_mask)
    test_dataset = ListFolder(root=opt.data_dir, image_paths=image_paths, labels_path=y, offset=testoffset, is_train=False, transform=test_transforms)
    trainloader = DataLoader(train_dataset, shuffle=False, drop_last=False, num_workers=opt.num_workers, batch_size=opt.train_batch_size, pin_memory=True)
    testloader = DataLoader(test_dataset, shuffle=False, drop_last=False, num_workers=opt.num_workers, batch_size=opt.test_batch_size, pin_memory=True)
    testiterator = iter(testloader)        

    logger.info('Length of dataloaders: Trainloader -> '+str(len(trainloader))+' Testloader -> '+str(len(testloader)))
    
    model.cuda()
    model.train()

#   model = torch.compile(model) # If using pytorch 2.0, uncomment this!
    if opt.fc_only:
        optimizer = SGD(model.fc.parameters(), lr=opt.maxlr, momentum=0.9, weight_decay=opt.weight_decay)
    else:
        optimizer = SGD(model.parameters(), lr=opt.maxlr, momentum=0.9, weight_decay=opt.weight_decay)
  
    end_of_stream = False
    for i, (images, target, mask_indices, _, train_idx) in enumerate(trainloader):

        # Evaluation Chunk
        if i%opt.num_gdsteps==0 and not end_of_stream:
            with torch.no_grad():
                model.eval()
                try:
                    test_images, test_targets, _,  sel_test_idx, _ = next(testiterator) #However you load the next iterator
                    test_images = test_images.cuda(non_blocking=True)
                except StopIteration:
                    print("Reached end of test set, Stop Training")
                    end_of_stream = True

                if opt.extract_feats:
                    model.temp_fc = model.fc.clone()
                    model.fc = torch.nn.Identity()
                    feats = model(test_images)
                    model.fc = model.temp_fc.clone()
                    output = model.fc(feats)    
                    featarr[sel_test_idx%opt.num_per_chunk] = feats.cpu().numpy()
                else:
                    output = model(test_images)

                predarr[sel_test_idx] = torch.argmax(output, dim=1).cpu().numpy()
                labelarr[sel_test_idx] = test_targets.numpy()
                
                if ((sel_test_idx.max()+1)//opt.num_per_chunk) > max_idx:
                    if opt.extract_feats: 
                        np.save(opt.log_dir+'/'+opt.exp_name+f'/features_{max_idx}.npy', featarr)
                        featarr = np.zeros((opt.num_per_chunk, opt.embed_size), dtype=np.float32)
                    np.save(opt.log_dir+'/'+opt.exp_name+'/preds.npy', predarr)
                    np.save(opt.log_dir+'/'+opt.exp_name+'/labels.npy', labelarr)
                    np.save(opt.log_dir+'/'+opt.exp_name+'/curr_idx.npy', np.array([train_idx.max()+trainoffset, sel_test_idx.max()], dtype=int)) #TODO: Check
                    save_model(opt=opt, model=model)
                    max_idx = max_idx+1 

        # Training Chunk
        model.train()    
        images, target = images.cuda(non_blocking=True), target.cuda(non_blocking=True)
        output = model(images)
 
        if isACE: # Do ACE correction on output logits
            mask_indices = mask_indices.cuda(non_blocking=True)
            mask   = torch.zeros_like(output)
            # unmask current classes
            present = target.unique()
            mask[:, present] = 1
            mask[mask_indices==0, :] = 1
            output  = output.masked_fill(mask == 0, -1e9)
        loss = nn.CrossEntropyLoss()(output, target)

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if opt.extract_feats: 
        np.save(opt.log_dir+'/'+opt.exp_name+f'/features_{max_idx}.npy', featarr)
    np.save(opt.log_dir+'/'+opt.exp_name+'/preds.npy', predarr)
    np.save(opt.log_dir+'/'+opt.exp_name+'/labels.npy', labelarr)
    np.save(opt.log_dir+'/'+opt.exp_name+'/curr_idx.npy', np.array([train_idx.max()+trainoffset, sel_test_idx.max()], dtype=int))
    save_model(opt=opt, model=model)
    return 


def ER_test(opt, model):
    model.eval()

    image_paths = opt.order_file_dir+'/test_image_paths.hdf5'
    y = opt.order_file_dir+'/test_labels.hdf5'

    if opt.extract_feats:
        model.temp_fc = model.fc.clone()
        model.fc = torch.nn.Identity()
    
    test_dataset = ListFolder(root=opt.data_dir, image_paths=image_paths, labels_path=y, offset=0, is_train=False, transform=test_transforms)
    testloader = DataLoader(test_dataset, shuffle=False, drop_last=False, num_workers=opt.num_workers, batch_size=opt.test_batch_size, pin_memory=True)

    labelarr, predarr = np.zeros(len(test_dataset.labels),dtype='u2'), np.zeros(len(test_dataset.labels),dtype='u2')
    if opt.extract_feats: featarr = np.zeros((len(test_dataset.labels),opt.embed_size),dtype=np.float32)

    with torch.no_grad():
        for i, (images, target, _, sel_test_idx, _) in enumerate(testloader):
            images = images.cuda(non_blocking=True)
            
            # compute output
            if opt.extract_feats:
                feats = model(images)
                output = model.temp_fc(feats)
                feats.cpu()
            else:
                output = model(images) 
            output = output.cpu()
            
            if opt.extract_feats: featarr[sel_test_idx] = feats.cpu().numpy()
            predarr[sel_test_idx] = torch.argmax(output, dim=1).cpu().numpy()
            labelarr[sel_test_idx] = target.numpy()
    
    np.save(opt.log_dir+'/'+opt.exp_name+'/test_preds.npy', predarr)
    np.save(opt.log_dir+'/'+opt.exp_name+'/test_labels.npy', labelarr)
    if opt.extract_feats: np.save(opt.log_dir+'/'+opt.exp_name+f'/test_features.npy', featarr)
    return

## Memory optimized code from the ACM repository

def train_acm_embedding(opt, model, ftmodel, logger):
    # Load pretraining data
    os.makedirs(opt.log_dir+'/'+opt.exp_name+'/', exist_ok=True)
    image_paths = opt.order_file_dir+'/pretrain_image_paths.hdf5'
    y = opt.order_file_dir+'/pretrain_labels.hdf5'
    train_routing_index_path = f'{opt.order_file_dir}/AdaptedACM.hdf5'
    train_dataset = ListFolder(root=opt.data_dir, image_paths=image_paths, labels_path=y, routing_idx_path=train_routing_index_path, offset=0, is_train=True, transform=train_transforms)
    trainloader = DataLoader(train_dataset, shuffle=False, drop_last=False, num_workers=opt.num_workers, batch_size=opt.train_batch_size, pin_memory=True)   
    logger.debug('Length of dataset: '+str(len(train_dataset)))
    logger.debug('Length of dataloader: '+str(len(trainloader)))
    model.cuda()
    model.eval()
    ftmodel.cuda()
    ftmodel.train()
    optimizer = SGD(ftmodel.parameters(), lr=opt.maxlr, momentum=0.9, weight_decay=opt.weight_decay)  
    scheduler = LinearLR(optimizer, T=opt.total_iterations)

    # Training loop
    for (images, targets, _, _, _) in trainloader:
        images, targets = images.cuda(non_blocking=True), targets.cuda(non_blocking=True)
        with torch.no_grad():
            feats = model(images)
        outputs = ftmodel(feats)  
        loss = torch.nn.CrossEntropyLoss()(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
    
    torch.save(ftmodel.state_dict(), opt.log_dir+'/'+opt.exp_name+'/ftmodel.pt')


def extract_features(opt, model, logger, ftmodel=None):
    # Load pretraining data
    os.makedirs(opt.log_dir+'/'+opt.exp_name+'/', exist_ok=True)
    image_paths = opt.order_file_dir+'/train_image_paths.hdf5'
    y = opt.order_file_dir+'/train_labels.hdf5'

    offset = opt.chunk_idx*opt.num_per_chunk
    dataset = ListFolder(root=opt.data_dir, image_paths=image_paths, labels_path=y, offset=offset, is_train=False, transform=test_transforms)
    loader = DataLoader(dataset, shuffle=False, drop_last=False, num_workers=opt.num_workers, batch_size=opt.test_batch_size, pin_memory=True)
    print('Length of dataloader: '+str(len(loader)))

    labelarr, featarr = np.zeros(len(y),dtype='u2'), np.zeros((opt.num_per_chunk, opt.embed_size),dtype=np.float32)
    if ftmodel is not None: predarr = np.zeros(len(y),dtype='u2')

    logger.info('==> Extracting features from idx '+str(offset)+'..')
    model.cuda()
    model.eval()

    if ftmodel is not None: 
        ftmodel.cuda()
        ftmodel.eval()

    # We will collect predictions, labels and features in corresponding numpy arrays
    with torch.inference_mode():
        for (image, label, _,  sel_idx, _) in loader:
            image = image.cuda(non_blocking=True)
            feat = model(image)
            if ftmodel is not None:
                feat = ftmodel.embed(feat)
                predprobs = ftmodel.fc(ftmodel.norm(feat))
                pred = torch.argmax(predprobs, dim=1)
                predarr[sel_idx%opt.num_per_chunk] = pred.cpu().numpy() 
            
            labelarr[sel_idx%opt.num_per_chunk] = label.cpu().numpy()
            featarr[sel_idx%opt.num_per_chunk] = feat.cpu().numpy()
        
            if ((sel_idx.max()+1)//opt.num_per_chunk) > opt.chunk_idx:
                np.save(opt.log_dir+'/'+opt.exp_name+f'/features_{opt.chunk_idx}.npy', featarr)
                if ftmodel is not None: np.save(opt.log_dir+'/'+opt.exp_name+f'/preds_{opt.chunk_idx}.npy', predarr)
                np.save(opt.log_dir+'/'+opt.exp_name+f'/labels_{opt.chunk_idx}.npy', labelarr)
    return

    
if __name__ == '__main__':
    # Parse arguments and init loggers
    torch.multiprocessing.set_sharing_strategy('file_system')
    opt = parse_args()

    opt.exp_name = f'{opt.dataset}_{opt.maxlr}_{opt.weight_decay}_{opt.sampler}_{opt.num_gdsteps}_{opt.train_batch_size}_{opt.test_batch_size}_{opt.model}_{opt.fc}_{opt.fc_only}_{opt.delay}_{opt.num_per_chunk}_{opt.cosine}'

    console_logger = get_logger(folder=opt.log_dir+'/'+opt.exp_name+'/')
    console_logger.info('==> Params for this experiment:'+str(opt))
    seed_everything(opt.seed)

    if opt.model == 'resnet50':
        model = models.resnet50(weights="IMAGENET1K_V2")
    elif opt.model == 'resnet50_I1B':
        model = torch.hub.load('facebookresearch/semi-supervised-ImageNet1K-models', 'resnet50_swsl')

    if opt.fc_only:
        for param in model.parameters():
            param.requires_grad = False
    
    if opt.mode=='ACM':
        ftmodel = EmbedLinearClassifier(dim=2048, embed_size=opt.embed_size, num_classes=opt.num_classes, cosfc=opt.cosine)
        if exists(opt.log_dir+'/'+opt.exp_name+'/ftmodel.pt'):
            ftmodel.load_state_dict(torch.load(opt.log_dir+'/'+opt.exp_name+'/ftmodel.pt'))
        else:
            train_acm_embedding(opt=opt, model=model, ftmodel=ftmodel, logger=console_logger)
        extract_features(opt=opt, model=model, ftmodel=ftmodel, logger=console_logger)
    else:
        model.fc = EmbedLinearClassifier(dim=2048, embed_size=opt.embed_size, num_classes=opt.num_classes, cosfc=opt.cosine)
        ER_train(opt=opt, model=model, image_paths=train_images_path, y=train_y_path, routing_index_path=train_routing_index_path, mask_values_path = mask_values_path, logger=console_logger, ftsize=opt.embed_size)
        ER_test(opt=opt, image_paths=test_images_path, image_labels=test_y_path, model=model)
