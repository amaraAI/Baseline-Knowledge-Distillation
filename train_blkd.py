import argparse
import numpy as np
import random 
import time
import warnings
import torchvision
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import MultiStepLR
import torch.nn.functional as F
from torchvision import datasets, transforms
from util.misc import CSVLogger
from model.resnet import ResNet18,ResNet34,ResNet50, ResNet101 , ResNet152
from torch.utils.tensorboard import SummaryWriter
        
model_options = ['resnet18','resnet34','resnet50','resnet101','resnet152', 'wideresnet','mobilenetv2','resnet8']
dataset_options = ['cifar100']

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed) 
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


parser = argparse.ArgumentParser(description='Training Baseline KD')
parser.add_argument('--model', '-a', default='resnet18',
                    choices=model_options)
parser.add_argument('--batch_size', type=int, default=128,
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=300,
                    help='number of epochs to train (default: 20)')
parser.add_argument('--learning_rate', type=float, default=0.1,
                    help='learning rate')
parser.add_argument('--data_augmentation', action='store_true', default=True,
                    help='augment data by flipping and cropping')
parser.add_argument('--seed', type=int, default=0,
                    help='random seed (default: 1)')
#parameters related to KD
parser.add_argument('--alphakd', type=float, default=0.7, help='alpha kd parameter')
parser.add_argument('--T', type= int , default=10, help='temperature for kd')
#parameters related to saving/resuming
parser.add_argument('--name', type=str, help='the name to give for the test_id(s)')
parser.add_argument('--resume', action='store_true', default=False,
                    help='resume from a certain checkpoit for training')


args = parser.parse_args()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
warnings.filterwarnings("ignore")

#this line is to create a name for each savefile
test_id =  args.name + args.model

#print the namespace.
print(args)
#set the seeds
set_seed(args.seed)

#verify if we need to resume... 
if args.resume: 
    ''' load the model '''
    checkpoint_path = 'blkd_checkpoints/' + test_id + '.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device)
    cnn.load_state_dict(checkpoint['model'])
    cnn_optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    start_epoch = checkpoint['epoch']+1
    best_acc = checkpoint['best_acc']
    best_epoch = checkpoint['best_epoch']

    ''' load the seeds states '''
    checkpoint_path = 'blkd_checkpoints_seeds/' + test_id + '.pt'
    checkpoint = torch.load(checkpoint_path, map_location=device) 
    torch.cuda.set_rng_state(checkpoint['cuda_rng_state'].cpu())
    torch.random.set_rng_state(checkpoint['torch_random_rng_state'].cpu())
    random.setstate(checkpoint['random_rng_state'])
    np.random.set_state(checkpoint['np_random_rng_state'])
else: 
    start_epoch = 0
    best_acc = 0.0
    best_epoch = 0



###########################################
###get the network of the teacher R152 ############

PATH_TEACH= "cifar100_resnet152.pt"

#######################################################################
##### LOADING THE TEACHER ########
netteach = ResNet152(num_classes=100)
netteach.load_state_dict(torch.load(PATH_TEACH,map_location=torch.device('cpu')))
netteach.to(device)
#################################################################################3
################################################################################


# summary for tensorboard
writer = SummaryWriter(log_dir='blkd_tb'+'/'+str(args.model)+'/'+str(args.seed)) # here it will create a folder with diff T's

#**********************************************************************************************************************
# ************ AT THIS STAGE WE ARE GOING TO USE THE SORTED VERSION OF THE TRAINING DATATSEt ***************

# Image Preprocessing

normalize = transforms.Normalize(mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                                     std=[x / 255.0 for x in [63.0, 62.1, 66.7]])
train_transform = transforms.Compose([])
if args.data_augmentation:
    train_transform.transforms.append(transforms.RandomCrop(32, padding=4))
    train_transform.transforms.append(transforms.RandomHorizontalFlip())
train_transform.transforms.append(transforms.ToTensor())
train_transform.transforms.append(normalize)

test_transform = transforms.Compose([
    transforms.ToTensor(),
    normalize])

num_classes = 100
    
trainset= datasets.CIFAR100(root='data/',
                                      train=True,
                                      transform=train_transform,
                                      download=True)

testset = datasets.CIFAR100(root='data/',
                                train=False,
                                transform=test_transform,
                                download=True)

# create the generators for each dataloader to seed
g = torch.Generator()
g.manual_seed(args.seed)
gv = torch.Generator()
gv.manual_seed(args.seed)

# create the dataloaders
trainloader = torch.utils.data.DataLoader(trainset,batch_size=args.batch_size,num_workers=4,worker_init_fn=seed_worker,generator=g)


val_loader = torch.utils.data.DataLoader(dataset=testset,
                                           batch_size=args.batch_size,
                                           shuffle=False,
                                           pin_memory=False,
                                           num_workers=4,
                                           worker_init_fn=seed_worker,generator=gv)

if args.model == 'resnet18':
    cnn = ResNet18(num_classes=num_classes)
elif args.model == 'wideresnet':
    cnn = WideResNet(depth=28, num_classes=num_classes, widen_factor=10,dropRate=0.3)
elif args.model == 'mobilenetv2':
    from model.mobilenetv2 import mobilenetv2
    cnn = mobilenetv2()

cnn = cnn.to(device)
criterion = nn.CrossEntropyLoss().to(device)
cnn_optimizer = torch.optim.SGD(cnn.parameters(), lr=args.learning_rate,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)


scheduler = MultiStepLR(cnn_optimizer, milestones=[50, 100, 150], gamma=0.2)
#below is the one using CRD 
#scheduler = MultiStepLR(cnn_optimizer, milestones=[150, 180, 210], gamma=0.1)

filename = 'jason_logs_blkd/' + test_id + '.csv'
csv_logger = CSVLogger(args=args, fieldnames=['epoch', 'train_acc', 'val_acc'], filename=filename)



def test(loader,cnn):
    cnn.eval()    # Change model to 'eval' mode (BN uses moving mean/var).
    correct = 0.
    total = 0.
    for images, labels in loader:
        images = images.float().to(device)
        labels = labels.long().to(device)

        with torch.no_grad():
            pred = cnn(images)
        pred = torch.max(pred.data, 1)[1]
        total += labels.size(0)
        correct += (pred == labels).sum().item()

    val_acc = correct / total
    cnn.train()
    return val_acc

def loss_fn_kd(outputs, labels, teacher_outputs, alpha, T):
    """
    Compute the knowledge-distillation (KD) loss given outputs, labels.
    "Hyperparameters": temperature and alpha
    """
    KD_loss = nn.KLDivLoss()(F.log_softmax(outputs/T, dim=1),
                             F.softmax(teacher_outputs/T, dim=1)) * (alpha * T * T) + \
              F.cross_entropy(outputs, labels) * (1. - alpha)

    return KD_loss
    
def train_epoch(dataloader,epoch, cnn, netteach, loss_fn_kd, cnn_optimizer):
    cnn.train()
    correct_avg = 0
    total_avg = 0 
    xkd_loss_avg = 0 

    for i, (im,l) in enumerate(dataloader):
        
        images = im.float().to(device)
        labels = l.squeeze().long().to(device)
        cnn.zero_grad()
        pred = cnn(images)
        with torch.no_grad():
            output_teacher = netteach(images)

        xkd_loss_avg=loss_fn_kd(pred, labels,output_teacher,args.alphakd, args.T)
        xkd_loss_avg.backward()
        cnn_optimizer.step()
        total_avg += labels.size(0)
        xkd_loss_avg += xkd_loss_avg.item()
        pred = torch.max(pred.data, 1)[1]
        correct_avg += (pred == labels.data).sum().item()
 
    return correct_avg / total_avg, xkd_loss_avg /total_avg


for epoch in range(start_epoch, args.epochs):
    start=time.time()
 
    train_acc, avg_loss = train_epoch(trainloader,epoch, cnn, netteach, loss_fn_kd, cnn_optimizer)
    
    writer.add_scalar('Train/Accuracy',train_acc,epoch)
    writer.add_scalar('Train/Loss',avg_loss,epoch)


    val_acc = test(val_loader,cnn)
    
    writer.add_scalar('Validation/Accuracy', val_acc, epoch)

    print("----------------------------------------------")
    print("EPOCH:", str(epoch))
    print('Val accuracy = ', val_acc)
    print("**************************************************")
    scheduler.step()     # Use this line for PyTorch >=1.4
    if best_acc < val_acc:
        torch.save(cnn.state_dict(), 'blkd_checkpoints/best/' + test_id + '.pt')
        best_acc = val_acc
        best_epoch = epoch
    print('best acc:{:.4f}, best epoch:{}'.format(best_acc, best_epoch))

    # save the model at the final epoch it could save 
    state_of_model = {
        'epoch': epoch,
        'model': cnn.state_dict(),
        'optimizer': cnn_optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'best_acc': best_acc,
        'best_epoch': best_epoch
            }
    torch.save(state_of_model,'blkd_checkpoints/' + test_id + '.pt')

    state_of_seeds = { 
            'torch_random_rng_state': torch.random.get_rng_state().cuda(),
            'cuda_rng_state': torch.cuda.get_rng_state().cuda(),
            'random_rng_state': random.getstate(),
            'np_random_rng_state': np.random.get_state(),
            }
    torch.save(state_of_seeds,'blkd_checkpoints_seeds/' + test_id + '.pt')

    row = {'epoch': str(epoch), 'train_acc': str(train_acc), 'val_acc' : str(val_acc)}
    csv_logger.writerow(row)

writer.close()
csv_logger.close()
