import os
import gc
import time
import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp
import random
import torch.backends.cudnn as cudnn
import numpy as np
from MMAE import MMAE
import json
from time import time
import datetime

from transformers import BertModel, AutoTokenizer
import torchvision.transforms as transforms
from torch.utils.data import Subset
from ETRI_Dataset import ETRI_Corpus_Dataset
from SWBD_Dataset import SWBD_Dataset
from HuBert import HuBert

from transformers import AutoTokenizer, AutoModelForPreTraining

class Trainer:
    def __init__(self, args) -> None:
        print(args)
        self.model = args.model
        self.num_workers = args.num_workers
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.world_size = args.world_size
        self.language = args.language

        self.seed = args.seed
        self.distributed = False
        self.rank = args.rank
        self.ngpus_per_node = torch.cuda.device_count()
        self.world_size = args.world_size * self.ngpus_per_node
        self.distributed = self.world_size > 1

        self.batch_size = int(self.batch_size / self.world_size)

        if os.environ.get("MASTER_ADDR") is None:
            os.environ["MASTER_ADDR"] = "localhost"
        if os.environ.get("MASTER_PORT") is None:
            os.environ["MASTER_PORT"] = "8888"

    def run(self):
        if self.distributed:
            mp.spawn(self._run, nprocs=self.world_size, args=(self.world_size,))
        else:
            self._run(0, 1)

    def _run(self, rank, world_size):

        self.local_rank = rank
        self.rank = self.rank * self.ngpus_per_node + rank
        self.world_size = world_size

        self.init_distributed()
        if self.seed is not None:
            random.seed(self.seed)
            np.random.seed(self.seed)
            torch.manual_seed(self.seed)
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed) # if use multi-GPU
            cudnn.deterministic = True
            cudnn.benchmark = False
            print('You have chosen to seed training. '
                'This will turn on the CUDNN deterministic setting, '
                'which can slow down your training considerably! '
                'You may see unexpected behavior when restarting '
                'from checkpoints.')
                
        
        if self.language == 'koBert':
            tokenizer = AutoTokenizer.from_pretrained('skt/kobert-base-v1')
            bert = BertModel.from_pretrained("skt/kobert-base-v1", add_pooling_layer=False, output_hidden_states=True, output_attentions=False)
            sentiment_dict = json.load(open('/data/minjae/BC/SentiWord_info.json', encoding='utf-8-sig', mode='r'))
            self.num_class = 4
        elif self.language == 'Bert':
            tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
            bert = BertModel.from_pretrained("bert-base-uncased", add_pooling_layer=False, output_hidden_states=True, output_attentions=False)
            sentiment_dict = {}
            self.num_class = 2
        else:
            raise NotImplementedError
        
        audio_model = HuBert()

        tf = transforms.ToTensor()

        if self.language == 'koBert':
            dataset = ETRI_Corpus_Dataset(path = 'balance', tokenizer=tokenizer, transform=tf, length=1.5)
        else :
            dataset = SWBD_Dataset(path = 'balance', tokenizer=tokenizer, length=1.5)
            
        self.train_dataset = Subset(dataset, range(0, int(len(dataset)*0.8)))
        self.val_dataset = Subset(dataset, range(int(len(dataset)*0.8), len(dataset)))

        self.train_sampler = torch.utils.data.distributed.DistributedSampler(self.train_dataset, shuffle=True, num_replicas=self.world_size, rank=self.rank)
        self.val_sampler = torch.utils.data.distributed.DistributedSampler(self.val_dataset, shuffle=False, num_replicas=self.world_size, rank=self.rank)

        self.train_dataloader = torch.utils.data.DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=False, sampler=self.train_sampler, num_workers=self.num_workers)
        self.val_dataloader = torch.utils.data.DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, sampler=self.val_sampler, num_workers=self.num_workers)

        self.model = MMAE(language_model=bert, audio_model=audio_model, num_class=self.num_class)
        
        self.model = self.model.to(self.local_rank)
        self.model_without_ddp = self.model
        if self.distributed:
            self.model = torch.nn.parallel.DistributedDataParallel(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

        language_model_params = []
        audio_model_params = []
        other_params = []
        
        for name, param in self.model.named_parameters():
            if 'language_model' in name:
                language_model_params.append(param)
            elif 'audio_model' in name:
                audio_model_params.append(param)
            else:
                other_params.append(param)
                
        adam_l_optimizer = torch.optim.Adam(language_model_params, lr=0.000001, weight_decay = 0.0001)  
        adam_a_optimizer = torch.optim.Adam(audio_model_params, lr=0.000001, weight_decay = 0.0001)    
        sgd_optimizer = torch.optim.Adam(other_params, lr=0.000001, weight_decay=0.001)                

        for epoch in range(20):
            start=time()
            loss = 0
            for b, batch in enumerate(self.train_dataloader):
                for key in batch:
                    batch[key] = batch[key].cuda()

                y = self.model(batch)
                # Get the logit from the model
                logit     = y["logit"]
                if self.is_MT:
                    sentiment = y["sentiment"]
                
                # Calculate the loss
                loss = 1 * self.model.pretext_forward(batch) + 0 * F.cross_entropy(logit, batch["label"])

                loss.backward()# Update the model parameters
                adam_l_optimizer.step()
                adam_a_optimizer.step()
                sgd_optimizer.step()

                # Zero the gradients
                adam_l_optimizer.zero_grad()
                adam_a_optimizer.zero_grad()
                sgd_optimizer.zero_grad()

#                print("Epoch : {}, Loss : {}".format(epoch, loss.item()))
                loss    += loss.item() * len(batch["label"])
            
                gc.collect()
            sec = time() -start
            times = str(datetime.timedelta(seconds=sec))
            short = times.split(".")[0]
                
            loss     /= len(self.train_dataloader)    
            print("Epoch : {}, Loss : {}, Time taken : {} ".format(epoch, loss, short))
                
        # Training loop
        
        for epoch in range(40):
            start=time()
            for b, batch in enumerate(self.train_dataloader):
                # Move the batch to GPU if CUDA is available
                for key in batch:
                    batch[key] = batch[key].cuda()

                y = self.model(batch)
                # Get the logit from the model
                logit     = y["logit"]

                # Calculate the loss
                loss_BC = F.cross_entropy(logit, batch["label"], reduction='none')

                loss = loss_BC
                
                loss = loss.mean()
                accuracy = (logit.argmax(dim=-1) == batch["label"]).float().mean()

                # Backpropagation
                loss.backward()

                # Update the model parameters
                adam_l_optimizer.step()
                adam_a_optimizer.step()
                sgd_optimizer.step()

                # Zero the gradients
                adam_l_optimizer.zero_grad()
                adam_a_optimizer.zero_grad()
                sgd_optimizer.zero_grad()

                l, c = logit.argmax(dim=-1).unique(return_counts=True)

                gc.collect()
            
            with torch.no_grad():
                # Validation loop
                accuracy = 0
                loss     = 0
                
                tp = torch.tensor([0 for _ in range(self.num_class)],device=self.local_rank)
                fp = torch.tensor([0 for _ in range(self.num_class)],device=self.local_rank)
                fn = torch.tensor([0 for _ in range(self.num_class)],device=self.local_rank)
                tn = torch.tensor([0 for _ in range(self.num_class)],device=self.local_rank)

                for batch in self.val_dataloader:
                    # Move the batch to GPU if CUDA is available
                    for key in batch:
                        batch[key] = batch[key].cuda()
                    y = self.model(batch)

                    # Get the logit from the model
                    logit     = y["logit"]
                    if self.is_MT:
                        sentiment = y["sentiment"]

                    loss_BC = F.cross_entropy(logit, batch["label"])
                    loss = loss_BC
                    
                    # Calculate the accuracy
                    accuracy += (torch.argmax(logit, dim=1) == batch["label"]).sum().item()
                    loss    += loss.item() * len(batch["label"])

                    # Calculate the confusion matrix
                    for i in range(len(batch["label"])):
                        for l in range(self.num_class):
                            if batch["label"][i] == l:
                                if logit.argmax(dim=-1)[i] == l:
                                    tp[l] += 1
                                else:
                                    fn[l] += 1
                            else:
                                if logit.argmax(dim=-1)[i] == l:
                                    fp[l] += 1
                                else:
                                    tn[l] += 1

                    if self.distributed:
                        dist.all_reduce(accuracy, op=dist.ReduceOp.SUM)
                        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
                        dist.all_reduce(tp, op=dist.ReduceOp.SUM)
                        dist.all_reduce(fp, op=dist.ReduceOp.SUM)
                        dist.all_reduce(fn, op=dist.ReduceOp.SUM)
                        dist.all_reduce(tn, op=dist.ReduceOp.SUM)
                        
                accuracy /= len(self.val_dataset)
                loss     /= len(self.val_dataset)
                precision = tp / (tp + fp)
                recall    = tp / (tp + fn)
                f1_score  = 2 * precision * recall / (precision + recall)
                score = f1_score.cpu().tolist()
                
                # Time check
                sec = time() -start
                times = str(datetime.timedelta(seconds=sec))
                short = times.split(".")[0]
                
                print("Epoch : {}, Accuracy : {}, Loss : {}".format(epoch, accuracy, loss))
                print("F1 score : {},  All : {}, Time taken : {} ".format(score, (score[0]*0.5+score[1]*0.41+score[2]*0.053+score[3]*0.037) , short))
            gc.collect()

    def init_distributed(self):
        if self.distributed:
            if torch.cuda.is_available():
                self.gpu    = self.local_rank % self.ngpus_per_nodes
                self.device = torch.device(self.gpu)
                if self.distributed:
                    self.local_rank = self.gpu
                    self.rank = self.node_rank * self.ngpus_per_nodes + self.gpu
                    time.sleep(self.rank * 0.1) # prevent port collision
                    print(f'rank {self.rank} is running...')
                    dist.init_process_group(backend=self.dist_backend, init_method=self.dist_url,
                                            world_size=self.world_size, rank=self.rank)
                    dist.barrier()
                    self.setup_for_distributed(self.is_main_process())
        else:
            self.device = torch.device('cpu')


    def is_main_process(self):
        return self.get_rank() == 0

    def setup_for_distributed(self, is_master):
        """
        This function disables printing when not in master process
        """
        import builtins as __builtin__
        builtin_print = __builtin__.print

        def print(*args, **kwargs):
            force = kwargs.pop('force', False)
            if is_master or force:
                builtin_print(*args, **kwargs)
        __builtin__.print = print

    def get_rank(self):
        if self.distributed:
            return dist.get_rank()
        return 0
    
    def get_world_size(self):
        if self.distributed:
            return dist.get_world_size()
        return 1
    