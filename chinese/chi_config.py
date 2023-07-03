

chi_args = dict()

#project structure
chi_args["code_path"] = '/home//Fed4CS' #absolute path to the code directory 
chi_args["recover"] = False #recover from path
chi_args["pretrained_model_file"] = None     #relative path to the pretrained model file 
# /pretrained/models/pretrained_model.pt

chi_args["trained_frontend_file"] = chi_args["code_path"] +'/transformer/visual_frontend.pt'
chi_args["dataset"] = "CCS"   #dataset used for training

# chi_args["data_path"] = "/mntnfs/med_data4/cs"   
chi_args["data_path"] = "/mntnfs/med_data5/CS_data_FL" #absolute path to the data directory
chi_args["ling_data_path"] = "/mntnfs/med_data5/CS_data_FL/fedlabels/hs_train_labels.txt"
chi_args["checkpoint_path"] = chi_args["code_path"] + "/checkpoints" #absolute path to the data directory
chi_args["store_name"] = '20230405fedavg' # name of log dir

#preprocessing
chi_args["roi_size"] = 64  #height and width of input greyscale lip region patch
chi_args["phone_to_index"] = {"b":1, "p":2, "m":3, "f":4, "d":5, "t":6, "n":7, "l":8, "g":9, "k":10, "h":11, "j":12,
                         "q":13, "x":14, "zh":15, "ch":16, "sh":17, "r":18, "z":19, "c":20, "s":21, "y":22, "w":23, "yu":24,
                         "a":25, "o":26, "e":27, "i":28, "u":29, "v":30, "ai":31, "ei":32, "ao":33, "ou":34, "er":35, "an":36,
                         "en":37, "ang":38, "eng":39, "ong":40, "-": 41, "<EOS>": 42}

chi_args["index_to_phone"] = {1:"b", 2:"p", 3:"m", 4:"f", 5:"d", 6:"t", 7:"n", 8:"l", 9:"g",10:"k", 11:"h", 12:"j",
                         13:"q", 14:"x", 15:"zh", 16:"ch", 17:"sh", 18:"r", 19:"z", 20:"c", 21:"s", 22:"y", 23:"w", 24:"yu",
                         25:"a", 26:"o", 27:"e", 28:"i", 29:"u", 30:"v", 31:"ai", 32:"ei", 33:"ao", 34:"ou", 35:"er", 36:"an",
                         37:"en", 38:"ang", 39:"eng", 40:"ong", 41:"-", 42: "<EOS>"} 

chi_args['blank'] = 0
chi_args['mask'] = 43
chi_args['spaceIx'] = chi_args["phone_to_index"]["-"]
# chi_args["<BOS>"] = chi_args["phone_to_index"]["<BOS>"]
chi_args["eosIx"] = chi_args["<EOS>"] = chi_args["phone_to_index"]["<EOS>"]


#transformer architecture
chi_args["n_trg_vocab"] = 43    #number of output characters for decoder
chi_args["src_pad_idx"] = 0
chi_args["d_src_seq"] = 512
chi_args["d_word_vec"] = 512
chi_args["d_model"]= 512
# chi_args["d_inner"]= 2048
chi_args["n_layers"] = 3 #3 for single
chi_args["n_layers_dec"] = 3
chi_args["n_head"] = 4
chi_args["d_k"] = 192
chi_args["d_v"]= 192
chi_args["dropout"] = 0.4
chi_args["n_position"] = 5000
chi_args["subsampling_dropout"] = 0.4
chi_args["ffd_dropout"] = 0.4
chi_args["ffd_expansion_factor"] = 2
chi_args["attn_dropout"] = 0.4
chi_args["conv_expansion_factor"] = 2
chi_args["conv_kernel_size"] = 31
chi_args["conv_dropout"] = 0.4
chi_args["half_step_residual"] = True

# dropout=0.4 for single

# semantic model architecture
chi_args["hidden_size"] = 512
chi_args["num_layer"] = 2

# semantic model training
chi_args["sem_lr"] = 1e-4 #1e-6 for Adam 1e-5 for SDG
chi_args["alpha"] = 0.005
chi_args["beta"] = 0.005
chi_args["gamma"] = 0.5

# chi_args["alpha"] = 0 # for fedavg
# chi_args["beta"] = 0
# chi_args["gamma"] = 0


# federated training
chi_args["glob_iters"] = 50 # number of global iterations
chi_args["num_clients"] = 4 # number of local clients
chi_args["frac"] = 1 # the fraction of clients
chi_args["local_epoches"] = 1 # epochs of local training
chi_args["global_ling_epoches"] = 10 # 10 # epochs of training global semantic model


#training
chi_args["single_speaker"] = False #True # single speaker and multiple speakers
chi_args['modal'] = 'lip_hand' # the modal used for trianing, options: lip, hand, or lip_hand
chi_args['cuda'] = True #True # if using cuda for training
chi_args['multi_gpus'] = False
chi_args["num_workers"] = 4 #dataloader num_workers argument
chi_args["seed"] = 123 #seed for random number generators
chi_args["batch_size"] = 1 #minibatch size
chi_args["step_size"] = 100  #number of steps # 50 for single
chi_args["print_freq"] = 10 #saving the model weights and loss/metric plots after every these many steps
chi_args["save_freq"] = 10

#optimizer and scheduler
chi_args["lr_mul"] = 0.05 #learning rate scaleer
chi_args["momentum1"] = 0.9 #optimizer momentum 1 value
chi_args["momentum2"] = 0.98   #optimizer momentum 2 value
chi_args["weight_decay"] = 0.06 # 0.05 for single
chi_args["warmup_steps"] = 5000 #5000 for single
chi_args["smoothing"] = True
chi_args["translate"] = False # evalution without labels


chi_args["lr_init"] = 0.0004
chi_args["lr_factor"] = 0.5
chi_args["patience"] = 5
chi_args["threshold"] = 0.01
chi_args["min_lr"]=1e-7

# decoder
chi_args["decodeScheme"] = "greedy"


#beam search
chi_args["BEAM_WIDTH"] = 15   #beam width
chi_args["LM_WEIGHT_ALPHA"] = 0.5  #weight of language model probability in shallow fusion beam scoring
chi_args["LENGTH_PENALTY_BETA"] = 0.3 #length penalty exponent hyperparameter
chi_args["THRESH_PROBABILITY"] = 0.0001 #threshold probability in beam search algorithm
chi_args["USE_LM"] = True  #whether to use language model for decoding


#testing
chi_args["TEST_DEMO_DECODING"] = "greedy"   #test/demo decoding type - "greedy" or "search"


if __name__ == "__main__":

    for key,value in chi_args.items():
        print(str(key) + " : " + str(value))