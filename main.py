import sys
import os
import torch
import numpy as np
import pandas as pd
from data_load import load_data_seq_pro, load_data_pdb_list, load_data_pdb_path, create_batches
from models import train_model2, predict_model
from collections import OrderedDict
import random
import timeit
import argparse
import shutil

# Add ProtVec utilities
def load_protvec_model(csv_file):
    """Load ProtVec model from CSV file."""
    print(f"Loading ProtVec model from {csv_file}")
    df = pd.read_csv(csv_file, sep='\t')
    protvec_dict = {}
    for index, row in df.iterrows():
        trigram = row.iloc[0]
        vector = row.iloc[1:].values.astype(float)
        protvec_dict[trigram] = vector
    print(f"ProtVec model loaded with {len(protvec_dict)} trigrams")
    return protvec_dict

def generate_protvec_embedding(sequence, protvec_dict):
    """Generate ProtVec embedding for a protein sequence."""
    # Generate trigrams
    sequence = sequence.upper()
    trigrams = [sequence[i:i+3] for i in range(len(sequence)-2)]
    
    # Find matching trigrams
    valid_trigrams = [t for t in trigrams if t in protvec_dict]
    
    if not valid_trigrams:
        return np.zeros(100)
    
    # Average trigram vectors
    return np.mean([protvec_dict[t] for t in valid_trigrams], axis=0)

def compute_sequence_protvec(sequence_dict, output_dir, protvec_dict):
    """Compute ProtVec embeddings for a set of sequences and save to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for name, seq in sequence_dict.items():
        output_file = os.path.join(output_dir, f"{name}.npy")
        
        # Skip if embedding already exists
        if os.path.exists(output_file):
            print(f"ProtVec embedding for {name} already exists")
            continue
        
        # Generate embedding
        embedding = generate_protvec_embedding(seq, protvec_dict)
        np.save(output_file, embedding)
        print(f"Generated ProtVec embedding for {name}")

def load_kidera_values(file_path, idx_to_amino):
    dict_aa_values = {}
    with open(file_path) as aa_kidera:
        for line in aa_kidera:
            line = line.strip().split(',')
            list_val = [float(v) for v in line[1:]]
            dict_aa_values[line[0]] = list_val
    values = np.array([np.array(dict_aa_values[aa]) for _, aa in idx_to_amino.items()])
    return torch.FloatTensor(values)


def train(save_dir, params, train_data_path, test_data_path):

    amino_acids = [aa for aa in 'ARNDCEQGHILKMFPSTWYV']
    aa_to_idx = {amino: index for index, amino in enumerate(amino_acids)}
    idx_to_aa = {index: amino for index, amino in enumerate(amino_acids)}
    kidera_embedding = load_kidera_values("kidera.csv", idx_to_aa)
    description = params["description"] + "_rsa"
    f_out = open(f"{save_dir}/res_file_b_cell_{description}.csv", "w")

    for i in range(1):
        params["description"] = f"{description}_{i}"
        params["predict_file"] = f"{save_dir}/predict_{description}_{i}.csv"
        train_data, invalid_proteins = load_data_seq_pro(train_data_path,params, test=False)
        random.shuffle(train_data)
        train_data, valid_data = train_data[:int(len(train_data) * 0.9)], train_data[int(len(train_data) * 0.9):]
        test_data, invalid_proteins = load_data_seq_pro(test_data_path, params, True)


        train_batches, list_aa_prop = create_batches(train_data,  params, aa_to_idx, train=True)
        val_batches, _ = create_batches(valid_data, params, aa_to_idx, list_aa_prop, False)
        test_batches, _ = create_batches(test_data,  params, aa_to_idx, list_aa_prop, False)

        test_auc, train_auc, val_auc = train_model2(train_batches, val_batches, test_batches, kidera_embedding,
                    params["epochs"], params["description"], save_dir, params, device, early_stopping=True)

        f_out.write(f"{test_auc},{val_auc},{train_auc}\n")


    return test_auc, train_auc, val_auc


def predict(params, save_dir):
    amino_acids = [aa for aa in 'ARNDCEQGHILKMFPSTWYV']
    aa_to_idx = {amino: index for index, amino in enumerate(amino_acids)}
    idx_to_aa = {index: amino for index, amino in enumerate(amino_acids)}

    list_aa_prop = []
    print("predicting 1...")
    with open(params["aa_prop_mean"]) as val:
        for line in val:
            line = line.strip().split(',')
            list_aa_prop.append([float(line[0]), float(line[1])])
    print("predicting 2...")

    if params["seq_input_path"]:
        test_data, invalid_proteins = load_data_seq_pro(params["seq_input_path"], params, False)
    elif params["pdb_id_list_path"]:
        test_data, invalid_proteins = load_data_pdb_list(params["pdb_id_list_path"], params)
    elif params["pdb_id_path"]:
        test_data, invalid_proteins = load_data_pdb_path(params["pdb_id_path"], params)
    else:
        print("No valid input path was entered")
        exit()

    kidera_embedding = load_kidera_values("kidera.csv", idx_to_aa)

    if model == "Boosting":
        params["model"] = "GCN"
        test_batches, _ = create_batches(test_data, params, aa_to_idx, list_aa_prop, False)
        dict_res_gcn = predict_model(test_batches, kidera_embedding, params)

        params["model"] = "BiLSTM"
        test_batches, _ = create_batches(test_data, params, aa_to_idx, list_aa_prop, False)
        dict_res_BiLSTM = predict_model(test_batches, kidera_embedding, params)

        dict_res = OrderedDict()

        for key, val in dict_res_BiLSTM.items():
            dict_res[key] = (val + dict_res_gcn[key])/2
    else:
        print("predicting 3...")

        test_batches, _ = create_batches(test_data, params, aa_to_idx, list_aa_prop, False)

        print("predicting 4...")
        dict_res = predict_model(test_batches, kidera_embedding, params)
                                     
    params["predict_file"] = f"{save_dir}/predict_{params['description']}.csv"
    f_predict = open(params["predict_file"], "w")
    for key, val in dict_res.items():
        pred =  1 if val >= params["threshold"] else 0
        f_predict.write(f"{key},{val},{pred}\n")

    f_predict.close()



if __name__ == '__main__':
    start = timeit.default_timer()
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--mode", type=str, required = True, choices = ["train", "predict"] )
    parser.add_argument("--init", type=str, required = True, choices = ["Kidera" ,"Random",  "Kidera+bio", "ESM-2", "ESM-IF1", "ProtVec"] )
    parser.add_argument("--model", type=str, required = True, choices = ["BiLSTM" ,"GCN",  "Boosting"] ) 
    parser.add_argument("--epi", type=str, default= "Linear",  required = True, choices = ["Linear", "Nonlinear", "Both"] )  
    parser.add_argument("--train_path", type=str, default= "None",  required = False )  
    parser.add_argument("--test_path", type=str, default= "None",  required = False ) 
    parser.add_argument("--test_seq_input", type=str,  required = False ) 
    parser.add_argument("--test_pdb_path", type=str,  required = False ) 
    parser.add_argument("--test_pdb_list", type=str,  required = False ) 
    parser.add_argument("--output_dir", type=str, default= "output",  required = False )
    
    args = parser.parse_args()
    mode = args.mode
    initialization = args.init
    print(initialization)
    if initialization not in ["Kidera+bio", "Random", "ESM-2", "ESM-IF1", "Kidera", "ProtVec"]:
        print(f"Initialization method is not valid. Please insert: Random/Kidera/Kidera+bio/ESM-2/ESM-IF1/ProtVec")
        exit()
    model = args.model #sys.argv[3]#"BiLSTM"#"BiLSTM"#sys.argv[3] #BiLSTM , GCN, boosting
    if model not in ["BiLSTM", "GCN", "Boosting"]:
        print(f"Model is not valid. Please insert: BiLSTM/GCN/Boosting")
        exit()

    protein_seq= args.epi #sys.argv[4]
    if protein_seq not in ["Linear", "Nonlinear", "Both"]:
        print(f"protein_seq is not valid. Please insert: Linear/Nonlinear")
        exit()

    train_path = args.train_path #sys.argv[5]#"
    test_path =  args.test_path #sys.argv[6]#"input/seq.fasta"
    seq_input_path = args.test_seq_input #sys.argv[7]

    params = {}
    if mode == "predict":
                models_dict = {"BiLSTM_ESM-IF1_Nonlinear" : "models/model_train_BiLSTM_ESM-IF1_nonlinear.pt", #_train_0
                               "GCN_ESM-IF1_Nonlinear" : "models/model_train_GCN_ESM-IF1_nonlinear.pt", #_train_4
                               "BiLSTM_ESM-2_Nonlinear" : "models/model_train_BiLSTM_ESM-2_nonlinear.pt", #_train_1
                               "GCN_ESM-2_Nonlinear" : "models/model_train_GCN_ESM-2_nonlinear.pt",  #_train_3
                               "BiLSTM_ESM-2_Linear" : "models/model_train_BiLSTM_ESM-2_Linear.pt", #_train_
                "BiLSTM_Random_Linear" : "models/model_train_BiLSTM_Random_linear.pt", #_train_,
                "BiLSTM_ESM-2_Linear" : "models/model_train_BiLSTM_ESM-2_linear.pt", #_train_
                "BiLSTM_Kidera_Linear" :  "models/model_train_BiLSTM_Kidera_linear.pt", #_train_
                 "BiLSTM_Kidera+bio_Linear" : "models/model_train_BiLSTM_Kidera+bio_linear.pt", #_train_
                 "BiLSTM_ESM-2_Both": "models/model_train_BiLSTM_ESM-2_Both.pt",
                 "BiLSTM_ProtVec_Linear": "models/model_train_BiLSTM_ProtVec_linear.pt",
            "GCN_ProtVec_Linear": "models/model_train_GCN_ProtVec_linear.pt",
            "BiLSTM_ProtVec_Nonlinear": "models/model_train_BiLSTM_ProtVec_nonlinear.pt",
            "GCN_ProtVec_Nonlinear": "models/model_train_GCN_ProtVec_nonlinear.pt",
            "BiLSTM_ProtVec_Both": "models/model_train_BiLSTM_ProtVec_Both.pt"  
                                   } 

                thresholds_dict = {"BiLSTM_ESM-IF1_Nonlinear" : 0.21, 
                                      "GCN_ESM-IF1_Nonlinear" : 0.21, 
                                       "GCN_ESM-2_Nonlinear" : 0.28,
                                      "BiLSTM_ESM-2_Nonlinear" : 0.29, 
                                      "BiLSTM_ESM-2_Linear" : 0.12,
                                      "BiLSTM_Random_Linear" : 0.1,

                                       "BiLSTM_ESM-2_Both" : 0.15,
                                        "BiLSTM_Kidera_Linear" : 0.12,
                 "BiLSTM_Kidera+bio_Linear" : 0.1,                       
                                      "Boosting_ESM-IF1_Nonlinear" :  0.3,
                                      
                                      "Boosting_ESM-2_Nonlinear" : 0.3,
                "BiLSTM_ProtVec_Linear": 0.12,
            "GCN_ProtVec_Linear": 0.12,
            "BiLSTM_ProtVec_Nonlinear": 0.21,
            "GCN_ProtVec_Nonlinear": 0.21,
            "BiLSTM_ProtVec_Both": 0.15,
            "Boosting_ProtVec_Linear": 0.3,
            "Boosting_ProtVec_Nonlinear": 0.3,
            "Boosting_ProtVec_Both": 0.3
                    }                                   
                """thresholds_dict = {"BiLSTM_ESM-IF1_Nonlinear" : 0.05202958360314369, 
                                      "GCN_ESM-IF1_Nonlinear" : 0.07649166136980057, 
                                      "BiLSTM_ESM-2_Nonlinear" : 0.07608705759048462, 
                                      "BiLSTM_ESM-2_Linear" : 0.08136602491140366,
                                      "BiLSTM_Random_Linear" : 0.016826564446091652,
                                      "BiLSTM_ESM-2_Linear": 0.022751489654183388,
                                       "BiLSTM_ESM-2_Both" : 0.024180497974157333,
                                        "BiLSTM_Kidera_Linear" : 0,
                 "BiLSTM_Kidera+bio_Linear" : 0,                       
                                      "Boosting_ESM-IF1_Nonlinear" :  0.038336802273988724,
                                      
                                      "Boosting_ESM-2_Nonlinear" : 0.03801000118255615,
                
                    }"""
                       
                bilstm_model, GCN_model = None, None
                params["threshold"] = thresholds_dict[f"{model}_{initialization}_{protein_seq}"]
                if not model == "GCN":
                  bilstm_model = models_dict[f"BiLSTM_{initialization}_{protein_seq}"]
            
                if not model == "BiLSTM":
                  GCN_model =  models_dict[f"GCN_{initialization}_{protein_seq}"]



                params["BiLSTM_model"] =  bilstm_model
                params["GCN_model"] = GCN_model

    params["train_path"] = train_path
    params["test_path"] = test_path
    params["seq_input_path"] = seq_input_path
    params["pdb_id_list_path"] = args.test_pdb_list
    params["pdb_id_path"] = args.test_pdb_path
    params["distb_dir"] = "input/dist_data/"
    params["aa_prop_mean"] = "input/val_to_normal.csv"
    params["epochs"] = 200
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    params["device"] = device
    save_dir = args.output_dir #"output_both"#sys.argv[5]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print(save_dir)

    i = 1000
    params["protvec_encoding_dir"] = "input/ProtVec_data"
    params["protvec_encoding_dir_tmp"] = "input/ProtVec_data_tmp"
    if not os.path.exists(params["protvec_encoding_dir_tmp"]):
        os.mkdir(params["protvec_encoding_dir_tmp"])
    params["protvec_model_path"] = "protVec_100d_3grams.csv"  # Path to your ProtVec model
    params["esmif1_encoding_dir"] = "input/ESMIF1_data_tmp"
    if not os.path.exists(params["esmif1_encoding_dir"]):
        os.mkdir(params["esmif1_encoding_dir"])
    params["esm2_encoding_dir"] = "input/ESM2_data_tmp"
    if not os.path.exists(params["esm2_encoding_dir"]):
        os.mkdir(params["esm2_encoding_dir"])
    params['dist_dir'] = "input/dist_data_tmp"
    if not os.path.exists(params["dist_dir"]):
        os.mkdir(params["dist_dir"])
    params["pdb_dir"] = "input/pdb_data_tmp"
    if not os.path.exists(params["pdb_dir"]):
        os.mkdir(params["pdb_dir"])


    params["model"] = model
    params["initialization"] = initialization
    if initialization == "ESM-2":
        params["embedding_dim"] = 1280
    elif initialization == "ESM-IF1":
        params["embedding_dim"] = 512
    elif initialization == "ProtVec":
        params["embedding_dim"] = 100
    else:
        params["embedding_dim"] = 10

    params["predict_file"] = "output/predict.csv"

    params["lstm_num_layer"] = 2
    params["encoding_dim"] = 10
    params["encoding_dim_lstm"] = 10
    params["w_d"] = 0.0001
    params["layer_size"] = 16
    params["i"] = i
    params["dist"] = 5
    params["dropout"] = 0.2
    params["lr"] = 0.0001
    if protein_seq != "Nonlinear":
        params["dropout"] = 0.25
        params["lr"] = 0.001
        params["encoding_dim_lstm"] = 100
        params["w_d"] = 1e-6
        
    input_for_dec = train_path.split('/')[-1].split('.')[0]
    params["description"] = f"{mode}_{model}_{initialization}_{input_for_dec}_{protein_seq}"

    if mode == "train":
        auc_test, auc_train, auc_valid = train(save_dir, params,  params["train_path"] ,  params["test_path"] )

    elif mode == "predict":
        predict(params, save_dir)
        

    shutil.rmtree("input/ESMIF1_data_tmp")
    shutil.rmtree("input/ESM2_data_tmp")
    shutil.rmtree("input/dist_data_tmp")
    shutil.rmtree("input/pdb_data_tmp")
    print("FINISH")
    print("time", timeit.default_timer() - start)


