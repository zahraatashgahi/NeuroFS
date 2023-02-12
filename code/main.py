# imports
import os; import sys; sys.path.append(os.getcwd())
import os.path
import numpy as np
import time

from utils import *
import datetime


from NeuroFS import NeuroFS

    
    
if __name__ == '__main__':
    import keras
    from keras import backend as K
    K.tensorflow_backend._get_available_gpus()
    #from keras import backend as K
    import tensorflow.compat.v1 as tf
    
    config = tf.ConfigProto( device_count = {'GPU': 1 , 'CPU': 2} ) 
    sess = tf.Session(config=config) 
    keras.backend.set_session(sess)

    print(tf.test.is_gpu_available() )# True/False
    print(tf.test.is_gpu_available(cuda_only=True) )

    #########################################################################
    ### parameter initialization and load data
    args = parse_arguments()
    args, data = load_data(args)
    
    import os
    os.environ['PYTHONHASHSEED']=str(args.seed)

    # 2. Set the `python` built-in pseudo-random generator at a fixed value
    import random
    random.seed(args.seed)

    # 3. Set the `numpy` pseudo-random generator at a fixed value
    import numpy as np
    np.random.seed(args.seed)

    # 4. Set the `tensorflow` pseudo-random generator at a fixed value
    #import tensorflow as tf
    tf.random.set_random_seed(args.seed)        
    if args.Kp is not None:
        args.K = int(args.Kp * data[0].shape[1])
        print(args.Kp, "% of features are going to be selected. ")
        print(args.Kp, "% = ", args.K, " features (#)", flush=True)
    elif args.K is not None:  
        args.Kp = round(args.K / data[0].shape[1], 2)
        print(args.Kp, "% of features are going to be selected. ")
        print(args.Kp, "% = ", args.K, " features (#)", flush=True)
    else:
        sys.exit("Error: You should inpout K or Kp ", flush=True)
    
    
    #########################################################################
    ### create directory to save the results
    
    path = "./results/"  + args.dataset_name+"/"+"K="+str(args.K)+"/"+ "seed="+str(args.seed) +"/"
    check_path(path)
    save_path = path #+ + "_" 
    args.save_path_weights = path +"/weights/" 
    check_path(args.save_path_weights)
    # save parameters in save_path
    f = open(save_path+"result.txt", 'w')
    args.save_path = save_path
    args.result_path = args.save_path+"result.txt"
    f.close() 
    # create a file to save log 
    print(save_path +"log.out")
    # redirect print
    import sys
    sys.stdout = open(save_path +"log.out", 'w')
    print("Feature Selection\n")
    print(args.Kp, "% of features are going to be selected. ")
    print(args.Kp, "% = ", args.K, " features (#)\n\n")   
    print(args)
    #log_file = open(save_path +"_log.txt", "w")  
    results = {"ACC_train_model":[], "Loss_train_model":[], 
                "ACC_test_model":[], "Loss_test_model":[], 
                "SVC":[], "KNN":[], "EXT":[],
                "CluACC":[], "NMI":[]}  
    args.results = results
    args.data = data
    
    
    #########################################################################
    ### Create Model
    model=NeuroFS(args)
    
        
    #########################################################################
    ### start training
    model.train(args)
    
