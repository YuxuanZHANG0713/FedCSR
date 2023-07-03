import torch
import os
import numpy as np

import copy
import torch.nn.functional as F
import time
import torch.nn as nn
import torch.optim as optim

from client import Client
from sklearn.metrics import confusion_matrix
from transformer.Models import CuedSpeechTransformer
from transformer.LSTM import Seq2Seq
from chinese.CCS_dataset import CCSDataset, LingDataset
from utils.general import num_params, train, evaluate
from utils.matrix import *

from tools import AverageMeter, accuracy
from losses import FCTCLoss

from sklearn.metrics import accuracy_score


class DiversityLoss(nn.Module):
    """
    Diversity loss for improving the performance.
    """
    def __init__(self, metric):
        """
        Class initializer.
        """
        super().__init__()
        self.metric = metric
        self.cosine = nn.CosineSimilarity(dim=2)

    def compute_distance(self, tensor1, tensor2, metric):
        """
        Compute the distance between two tensors.
        """
        if metric == 'l1':
            return torch.abs(tensor1 - tensor2).mean(dim=(2,))
        elif metric == 'l2':
            return torch.pow(tensor1 - tensor2, 2).mean(dim=(2,))
        elif metric == 'cosine':
            return 1 - self.cosine(tensor1, tensor2)
        else:
            raise ValueError(metric)

    def pairwise_distance(self, tensor, how):
        """
        Compute the pairwise distances between a Tensor's rows.
        """
        n_data = tensor.size(0)
        tensor1 = tensor.expand((n_data, n_data, tensor.size(1)))
        tensor2 = tensor.unsqueeze(dim=1)
        return self.compute_distance(tensor1, tensor2, how)

    def forward(self, noises, layer):
        """
        Forward propagation.
        """
        if len(layer.shape) > 2:
            layer = layer.view((layer.size(0), -1))
        layer_dist = self.pairwise_distance(layer, how=self.metric)
        noise_dist = self.pairwise_distance(noises, how='l2')
        return torch.exp(torch.mean(-noise_dist * layer_dist))


class Server:
    def __init__(self, cfg):
        # Set up the main attributes
        self.cfg = cfg # arguments from CS dataset
        self.num_glob_iters = self.cfg['glob_iters'] # number of global iterations
        self.early_stop = 50 #early stop epochs for client local train
        self.num_clients = self.cfg['num_clients'] # number of clients
        self.frac_clients = max(int(self.cfg['frac'] * self.num_clients), 1) # number of clients selected per round

        self.recover = self.cfg['recover']

        self.glob_dataset_test = CCSDataset(self.cfg["data_path"], is_train = False, phone2index=self.cfg["phone_to_index"], client = 'multi_speaker')

        self.eval_loader = torch.utils.data.DataLoader(self.glob_dataset_test, batch_size=1, shuffle=False, num_workers=self.cfg['num_workers'], pin_memory=True, drop_last=True)


        self.init_loss_fn()
        self.init_model()
        self.init_client()
        


        self.root_log = 'log'
        self.checkpoints = 'checkpoints'
        self.store_name = self.cfg['store_name']

        self.prepare_folders()
        self.log = open(os.path.join(self.root_log, self.store_name, 'log.csv'), 'w')

        # components added for fed KD

        self.ling_dataset = LingDataset(self.cfg['phone_to_index'], self.cfg['ling_data_path'])
        self.ling_dataloader = torch.utils.data.DataLoader(self.ling_dataset, batch_size=1, shuffle=True, num_workers=self.cfg['num_workers'], pin_memory=True, drop_last=True)
        # self.optimizer = optim.SGD(self.glob_ling_model.parameters(), lr = self.cfg["sem_lr"], weight_decay=1e-4) 
        self.optimizer = optim.Adam(self.glob_ling_model.parameters(), lr = self.cfg["sem_lr"])#, weight_decay=1e-4) 
        self.code_book = [i for i in range(len(self.cfg["phone_to_index"])+1)]

    def prepare_folders(self):
        folders_util = [self.root_log, self.checkpoints,
                        os.path.join(self.root_log, self.store_name),
                        os.path.join(self.checkpoints, self.store_name)]
        for folder in folders_util:
            if not os.path.exists(folder):
                print('creating folder ' + folder)
                os.mkdir(folder)


    def init_model(self):
        # self.glob_model = models.__dict__[self.model_name](num_classes=self.num_classes)

        # if self.args.gpu is not None:
        #     torch.cuda.set_device(self.args.gpu)
        #     self.glob_model = self.glob_model.cuda(self.args.gpu)
        # else:
        #     # DataParallel will divide and allocate batch_size to all available GPUs
        #     self.glob_model = torch.nn.DataParallel(self.glob_model).cuda()
        self.glob_model = CuedSpeechTransformer(
            dataset = 'Chinese',
            d_src_seq=self.cfg["d_word_vec"],
            n_trg_vocab=self.cfg["n_trg_vocab"],
            d_k=self.cfg["d_k"],
            d_v=self.cfg["d_v"],
            d_model=self.cfg["d_model"],
            d_word_vec=self.cfg["d_word_vec"],
            n_layers=self.cfg["n_layers"],
            n_layers_dec = self.cfg["n_layers_dec"],
            n_head=self.cfg["n_head"],
            n_position=self.cfg["n_position"],
            subsampling_dropout = self.cfg["subsampling_dropout"],
            ffd_dropout=self.cfg["ffd_dropout"],
            ffd_expansion_factor=self.cfg["ffd_expansion_factor"],
            attn_dropout=self.cfg["attn_dropout"],
            conv_expansion_factor=self.cfg["conv_expansion_factor"],
            conv_kernel_size=self.cfg["conv_kernel_size"],
            conv_dropout=self.cfg["conv_dropout"],
            half_step_residual=self.cfg["half_step_residual"],
           )

        # self.glob_ling_model = LingLSTM(n_trg_vocab = self.cfg["n_trg_vocab"], 
        #                                 d_words= self.cfg["d_word_vec"], 
        #                                 hidden_size= self.cfg["hidden_size"], 
        #                                 num_layers=self.cfg["num_layer"])

        self.glob_ling_model = Seq2Seq(n_trg_vocab = self.cfg["n_trg_vocab"], 
                                        d_words= self.cfg["d_word_vec"], 
                                        hidden_size= self.cfg["hidden_size"], 
                                        num_layers=self.cfg["num_layer"])

        self.device = torch.device('cuda:0' if self.cfg['cuda'] else 'cpu')
        if self.recover:
            self.load_model()


        # if self.cfg["multi_gpus"] is False and self.cfg['cuda']:
        #     self.glob_model.to(device)
        # elif self.cfg["multi_gpus"] is True and self.cfg['cuda']:
        #     torch.nn.DataParallel(self.glob_model).cuda()
        # else:
        #     print("using cpu for training!")
        #     self.glob_model=self.glob_model.to(device)

  
    def init_client(self):
        self.clients = []
       
        for i in range(self.num_clients):
            client = Client(self.cfg, i, self.glob_model, self.glob_ling_model)

            self.clients.append(client)

        print("finish initizing clients!")
        
    def send_parameters(self, mode='all', beta=1, selected=True):
        global_parameters_model = {}
        global_parameters_ling = {}
        for key, value in self.glob_model.state_dict().items():
            global_parameters_model[key]=value.clone()
        for key, value in self.glob_ling_model.state_dict().items():
            global_parameters_ling[key]=value.clone()

        clients = self.clients
        if selected:
            assert (self.selected_clients is not None and len(self.selected_clients) > 0)
            clients = self.selected_clients

        for client in clients:
            if mode == 'all': # share all parameters 
                client.set_parameters(global_parameters_model, global_parameters_ling, beta=beta)
            # else: # share only subset of parameters
            #     client.set_shared_parameters(self.model,mode=mode)


    def aggregate_parameters(self):
        assert (self.selected_clients is not None and len(self.selected_clients) > 0)

        glob_state_dict = None

        total_train = 0
        for client in self.selected_clients:
            total_train += client.train_samples

        for client in self.selected_clients:
            ratio = client.train_samples / total_train
            local_state = client.model.state_dict()

            if glob_state_dict == None:
                glob_state_dict = {}
                for key in local_state.keys():
                    if 'num_batches_tracked' in key:
                        glob_state_dict[key] = torch.zeros_like(local_state[key]) + int(local_state[key].clone()*ratio)
                    else:
                        glob_state_dict[key] = local_state[key].clone()*ratio
            else:
                for key in local_state.keys():
                    if 'num_batches_tracked' in key:
                        glob_state_dict[key] += int(local_state[key].clone()*ratio)
                    else:
                        glob_state_dict[key] += local_state[key].clone()*ratio

        self.glob_model.load_state_dict(glob_state_dict)


    def save_model(self):
        model_path = os.path.join("checkpoints")
        if not os.path.exists(model_path):
            os.makedirs(model_path)
        torch.save(self.glob_model, os.path.join(model_path, self.store_name + "_server_" + ".pt"))
        torch.save(self.glob_ling_model, os.path.join(model_path, self.store_name + "_ling_" + ".pt"))


    def load_model(self):
        model_path = os.path.join(model_path, self.store_name + "_server_" + ".pt")
        ling_model_path = os.path.join(model_path, self.store_name + "_ling_" + ".pt")
        assert (os.path.exists(model_path) and os.path.exists(ling_model_path))
        self.glob_model = torch.load(model_path)
        self.glob_ling_model = torch.load(ling_model_path)

    
    def select_clients(self, frac_clients, return_idx=False):
        '''selects num_clients clients weighted by number of samples from possible_clients
        Args:
            num_clients: number of clients to select; default 20
                note that within function, num_clients is set to
                min(num_clients, len(possible_clients))
        Return:
            list of selected clients objects
        '''
        if(frac_clients == len(self.clients)):
            print("All clients are selected")
            return self.clients, [i for i in range(frac_clients)]

        frac_clients = min(frac_clients, len(self.clients))

        if return_idx:
            cleint_idxs = np.random.choice(range(len(self.clients)), frac_clients, replace=False)
            return [self.clients[i] for i in cleint_idxs], cleint_idxs
        else:
            return np.random.choice(self.clients, frac_clients, replace=False)


    def init_loss_fn(self):
        self.loss_CTC = FCTCLoss(blank=self.cfg['blank'], zero_infinity=False)
        self.nllloss=nn.NLLLoss()
        # self.ensemble_loss=nn.KLDivLoss(log_target=True, reduction='batchmean')
        self.ensemble_loss=nn.MSELoss(reduction='mean')

        # self.cosine_loss=nn.CosineEmbeddingLoss(reduction='mean')

        # self.ce_loss = nn.CrossEntropyLoss()
        # self.diversity_loss = DiversityLoss(metric='l1')


    def save_results(self, result):
        self.log.write(result + '\n')
        self.log.flush()

    def train_ling_model(self):
        print('Training semantic model...')
        glob_model_head = copy.deepcopy(self.glob_model.head)
        glob_model_head.sub_rate = 1 # decode does not subsample
        glob_model_head.to(self.device)
        self.glob_ling_model.code_emb.load_state_dict(glob_model_head.code_emb.state_dict())
        self.glob_ling_model.to(self.device)
        # self.glob_ling_model.head.sub_rate = 1
        for iter in range(self.cfg['global_ling_epoches']): 
            total_loss = 0
            total_loss_sim = 0
            total_loss_CE = 0
            total_acc = 0
            total_len = 0
            for i, seq in enumerate(self.ling_dataloader):
                self.optimizer.zero_grad()
                self.glob_ling_model.train()
                seq = seq.to(self.device)
                # output = self.glob_ling_model(seq)
                feature, output = self.glob_ling_model(seq, seq, seq.shape[1])

                with torch.no_grad():
                    emb = copy.deepcopy(self.glob_ling_model.code_emb)
                    emb.to(self.device)
                    code_book = torch.tensor(self.code_book).to(self.device)
                    input_len = torch.tensor(len(seq)).int().to(self.device)
                    _, _, output_ling_feature, output_ling, output_post, _ = glob_model_head(emb(seq), emb(seq), input_len, code_book)
                    output_ling_feature = output_ling_feature.transpose(0,1)
                    # output_post = output_post.transpose(0,1)
                
                #KLloss
                feature = feature.reshape(-1, feature.shape[2])
                output = output.reshape(-1, output.shape[2])
                output_ling_feature = output_ling_feature.reshape(-1, output_ling_feature.shape[2])

                loss_sim = self.ensemble_loss(output_ling_feature, feature)
                # loss_sim = self.ensemble_loss(F.log_softmax(output_ling_feature, dim=-1), F.log_softmax(feature, dim=-1))
       
                #CEloss 
                gt = seq.reshape(-1)
                loss_NLL = self.nllloss(output, gt)

                loss = loss_NLL + self.cfg['alpha'] * loss_sim #TODO: add parameter
                # print('lossnnl:', loss_NLL, 'losssim:', loss_sim)
                loss.backward()
                # loss_NLL.backward()
                self.optimizer.step()

                # acc
                pred = torch.argmax(output, dim=1)
                acc = accuracy_score(gt.flatten().tolist(), pred.flatten().tolist())
                seq_len = len(gt)


                total_loss += loss
                total_loss_sim += loss_sim
                total_loss_CE += loss_NLL
                total_acc += acc*seq_len
                total_len += seq_len

                #TODO: KLdivoutput NaN
            print('epoch:', iter,' total loss:', total_loss, ' total loss sim:', total_loss_sim, ' total loss CE:', total_loss_CE, 'total acc: ', total_acc/total_len)





    def train(self):

        best_cer = 1.0
        for glob_iter in range(self.num_glob_iters):

            print("-------------Global iter: ", glob_iter, " -------------")

            #send trained embedding to glob_model
            self.glob_model.head.code_emb.load_state_dict(self.glob_ling_model.code_emb.state_dict())

            # training local models

            self.selected_clients, self.client_idxs = self.select_clients(self.frac_clients, return_idx=True)
            
            self.send_parameters()# broadcast averaged prediction model

            chosen_verbose_user = np.random.randint(0, len(self.clients))

            self.timestamp = time.time() # log user-training start time

            for idx, (client_id, client) in enumerate(zip(self.client_idxs, self.selected_clients)): # allow selected clients to train
                print("-----Client ID: ", idx, " -----")

                verbose = client_id == chosen_verbose_user

                # perform regularization using generated samples after the first communication round
                client.local_train(client_id, glob_iter, early_stop=self.early_stop, verbose = verbose and glob_iter > 0, regularization= glob_iter < 0 )
                

            curr_timestamp = time.time() # log  user-training end time
            train_time = (curr_timestamp - self.timestamp) / len(self.selected_clients)

            self.timestamp = time.time() # log server-agg start time
            self.aggregate_parameters()
            curr_timestamp=time.time()  # log  server-agg end time
            agg_time = curr_timestamp - self.timestamp
 
            # self.save_model()

            # self.timestamp = time.time() # log server-generator start time
            # self.train_generator(self.gen_batchsize, epoches=self.gen_epoches // self.n_teacher_iters, verbose=True)
            # curr_timestamp=time.time()  # log  server-agg end time
            # gen_time = curr_timestamp - self.timestamp

            # if glob_iter  > 0 and glob_iter % 20 == 0 and self.latent_layer_idx == 0:
            #     self.visualize_images(self.generative_model, glob_iter, repeats=10)
           
            valCER = self.evaluate()
            if valCER <= best_cer:
                self.save_model()
                best_cer = valCER

            # training semantic model
            self.train_ling_model()
            
            
            
            
        # self.save_results(result)

        # if best_acc > glob_acc:
        #     self.save_model()


    def evaluate(self):

        device = torch.device('cuda:0' if self.cfg['cuda'] else 'cpu')
        start_time = time.time()
        validationLoss, validationCER, validationWER, predictions, targets = evaluate(self.glob_model, self.eval_loader, self.loss_CTC, device, return_result=True, args=self.cfg)
        output = ("Global: Val.Loss: %.6f ||Val.CER: %.3f ||Val.WER: %.3f || time: %d"
              %(validationLoss, validationCER, validationWER, time.time()-start_time))  # TODO
        print(output)

        return validationCER

        # self.glob_model.eval()
        # batch_time = AverageMeter('Time', ':6.3f')
        # losses = AverageMeter('Loss', ':.4e')
        # top1 = AverageMeter('Acc@1', ':6.2f')
        # top5 = AverageMeter('Acc@5', ':6.2f')
        
        # # switch to evaluate mode
        # all_preds = []
        # all_targets = []
        # with torch.no_grad():
        #     end = time.time()
        #     for i, (input, target) in enumerate(self.eval_loader):
        #         if self.args.gpu is not None:
        #             input = input.cuda(self.args.gpu, non_blocking=True)
        #         target = target.cuda(self.args.gpu, non_blocking=True)

        #         # compute output
        #         output = self.glob_model(input)
        #         loss = self.ce_loss(output, target)

        #         # measure accuracy and record loss
        #         acc1, acc5 = accuracy(output, target, topk=(1, 5))
        #         losses.update(loss.item(), input.size(0))
        #         top1.update(acc1[0], input.size(0))
        #         top5.update(acc5[0], input.size(0))

        #         # measure elapsed time
        #         batch_time.update(time.time() - end)
        #         end = time.time()

        #         _, pred = torch.max(output, 1)
        #         all_preds.extend(pred.cpu().numpy())
        #         all_targets.extend(target.cpu().numpy())

               
        #         output = ('Test: [{0}/{1}]\t'
        #                 'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
        #                 'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
        #                 'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        #                 'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
        #             i, len(self.eval_loader), batch_time=batch_time, loss=losses,
        #             top1=top1, top5=top5))
        #     print(output)
        #     cf = confusion_matrix(all_targets, all_preds).astype(float)
        #     cls_cnt = cf.sum(axis=1)
        #     cls_hit = np.diag(cf)
        #     cls_acc = cls_hit / cls_cnt
        #     output = ('Eval Results: Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Loss {loss.avg:.5f}'
        #             .format(top1=top1, top5=top5, loss=losses))
        #     out_cls_acc = 'Eval Class Accuracy: %s'%((np.array2string(cls_acc, separator=',', formatter={'float_kind':lambda x: "%.3f" % x})))
        #     print(output)
        #     print(out_cls_acc)
        #     # if log is not None:
        #     #     log.write(output + '\n')
        #     #     log.write(out_cls_acc + '\n')
        #     #     log.flush()

        #     # tf_writer.add_scalar('loss/test_'+ flag, losses.avg, epoch)
        #     # tf_writer.add_scalar('acc/test_' + flag + '_top1', top1.avg, epoch)
        #     # tf_writer.add_scalar('acc/test_' + flag + '_top5', top5.avg, epoch)
        #     # tf_writer.add_scalars('acc/test_' + flag + '_cls_acc', {str(i):x for i, x in enumerate(cls_acc)}, epoch)

        # return top1.avg, losses.avg



    

