import os
import pathlib
import csv
import esm as esmif1
import numpy as np
import pandas as pd

import torch
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from esm_embbeding import esm_embbeding
from download_pdb import extract_sequence_from_pdb, download_pdb_and_save
from torch_geometric.data import Data
from torch_sparse import SparseTensor


import re
regex = re.compile('[ARNDCEQGHILKMFPSTWYV]+$')


ESM_IF1_PATH = "input/ESMIF1_data"
ESM_2_PATH = "input/ESM2_data"
DIST_PATH = "input/dist_data"
PDB_PATH = "input/pdb_data"

# Add these functions at the top with other similar functions
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

def compute_protvec_embeddings(name_pro, protein, protvec_dict, protvec_dir):
    """Compute ProtVec embedding for a protein sequence and save it."""
    # Generate trigrams
    sequence = protein.upper()
    trigrams = [sequence[i:i+3] for i in range(len(sequence)-2)]
    
    # Find matching trigrams
    valid_trigrams = [t for t in trigrams if t in protvec_dict]
    
    if not valid_trigrams:
        # Handle case where no trigrams match
        embedding = np.zeros(100)
    else:
        # Average trigram vectors
        embedding = np.mean([protvec_dict[t] for t in valid_trigrams], axis=0)
    
    # Convert to tensor and save
    embedding_tensor = torch.tensor(embedding, dtype=torch.float)
    torch.save(embedding_tensor, f"{protvec_dir}/{name_pro}.pt")
    
    return embedding_tensor

def load_data_seq_pro(path, params, test = False):
    all_data = []
    invalid_proteins = []
    with open(path, mode='r') as infile:
        for line in infile:
            if ">" in line:
                name_pro = line.strip().replace('>', "")###.replace("ID_", "")
            else:
                protein = line.strip()
                list_tag = [1 if aa.isupper() else 0 for aa in protein]
                protein = protein.upper()
                if regex.match(protein):
                    if params["initialization"] == "ESM-IF1" or params["model"] == "GCN":
                        if not pathlib.Path(f"{PDB_PATH}/{name_pro}.pdb").is_file():
                            status = download_pdb_and_save(name_pro, params["pdb_dir"])
                            if status == 0 :
                                pdb_id, chain = name_pro.split("_")
                                dict_chain_seq = extract_sequence_from_pdb(f"{params['pdb_dir']}/{pdb_id}.pdb", pdb_id, params['pdb_dir'], params['dist_dir'], chain_pdb = chain )
                                if not dict_chain_seq[f"{chain}_0"] == protein:
                                    invalid_proteins.append(name_pro)
                                    continue
                            else:
                                invalid_proteins.append(name_pro)
                                continue
                    print(f"{name_pro},{len(protein)},{len(list_tag)}")
                    all_data.append((name_pro, protein, list_tag))


    print(len(all_data))
    return all_data, invalid_proteins


def load_data_pdb_list(path, params):
    all_data = []
    invalid_proteins = []
    with open(path, mode='r') as infile:
        for line in infile:
            pdb_id = line.strip()
            status = download_pdb_and_save(pdb_id, params["pdb_dir"])
            if status == 0:
                dict_chain_seq = extract_sequence_from_pdb(f"{params['pdb_dir']}/{pdb_id}.pdb", pdb_id, params['pdb_dir'], params['distb_dir'])
                for chain, seq in dict_chain_seq.items():
                    all_data.append((chain, seq, []))
            else:
                invalid_proteins.append(pdb_id)
                continue

        print(len(all_data))
        return all_data, invalid_proteins


def load_data_pdb_path(path, params):
    all_data = []
    invalid_proteins = []
    all_data_names = set()
    for file in os.listdir(path):
        filename = os.fsdecode(file)
        print(file)
        if filename.endswith(".pdb"):
            pdb_id = filename.strip()
            dict_chain_seq = extract_sequence_from_pdb(f"{path}/{filename}", pdb_id, params['pdb_dir'], params['distb_dir'])
            for chain, seq in dict_chain_seq.items():
                all_data.append((chain, seq, []))

    print(len(all_data))
    return all_data, invalid_proteins

def call_esmif1(name_pro, esmif1_encoding_dir, pdb_dir):

    model, alphabet = esmif1.pretrained.esm_if1_gvp4_t16_142M_UR50()
    ## batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    target_chain_id = name_pro.split("_")[1]
    chain_ids = [target_chain_id]  # ['A']

    structure = esmif1.inverse_folding.util.load_structure(f"{pdb_dir}/{name_pro}.pdb",
                                                           chain_ids)
    
    coords, native_seqs = esmif1.inverse_folding.multichain_util.extract_coords_from_complex(structure)

    rep = esmif1.inverse_folding.multichain_util.get_encoder_output_for_complex(model, alphabet, coords,
                                                                                target_chain_id)
    torch.save(
        rep,
        f"{esmif1_encoding_dir}/{name_pro}.pt",
    )

    return rep

def call_esm2(name_pro, protein, esm_encoding_dir ):
    print("in")
    accs = [name_pro]
    seqs = [protein]
    esm2 = esm_embbeding(accs, seqs, esm_encoding_dir)
    esm2.create_fasta_for_esm_transformer()
    esm2.call_esm_script()
    rep = esm2.prepare_esm_data()[0]

    torch.save(
        rep,
        f"{esm_encoding_dir}/{name_pro}.pt",
    )

    return rep

def batch_call_esm2(protein_data, params):
    """
    Recibe protein_data: lista de tuplas (name_pro, protein, list_tag) 
    para las cuales no existe embedding y las procesa conjuntamente.
    """
    names = [name for name, protein, _ in protein_data]
    seqs = [protein for name, protein, _ in protein_data]
    # Se crea una única instancia de esm_embbeding para el batch
    esm2_inst = esm_embbeding(names, seqs, params["esm2_encoding_dir"])
    esm2_inst.create_fasta_for_esm_transformer()
    esm2_inst.call_esm_script()
    reps = esm2_inst.prepare_esm_data()  # Se asume que regresa una lista de representaciones
    # Guardamos los embeddings
    for name, rep in zip(names, reps):
        torch.save(rep, f"{ESM_2_PATH}/{name}.pt")

def calc_properties_of_amino_acids(seq):
    list_prop = []
    analysed_seq = ProteinAnalysis(seq)
    list_prop.append(analysed_seq.molecular_weight() )
    list_prop.append(analysed_seq.gravy())
    list_prop.append(analysed_seq.secondary_structure_fraction()[0])
    list_prop.append(analysed_seq.aromaticity() )
    list_prop.append(analysed_seq.instability_index() )
    list_prop.append(analysed_seq.isoelectric_point() )
    molar_reduced, molar_disulfid = analysed_seq.molar_extinction_coefficient()
    list_prop.append(molar_disulfid )
    list_prop.append(molar_reduced )

    return list_prop


def convert_pep_to_list_of_aa(pep, amino_to_idx):
    list_pep = []
    for i in range(len(pep)):
        amino = pep[i]
        list_pep.append(amino_to_idx[amino])
    return list_pep


# Then modify the create_batches function to handle ProtVec
def create_batches(data, params, amino_to_idx, list_aa_prop=None, train=False):
    batches_list = []
    dict_bio_prop = {i: [] for i in range(8)}

    # Process proteins that need ProtVec embeddings
    if params["initialization"] == "ProtVec":
        missing_protvec = []
        # Load ProtVec model if needed
        if not hasattr(params, "protvec_dict"):
            params["protvec_dict"] = load_protvec_model(params["protvec_model_path"])
        
        # Check which proteins need processing
        for name_pro, protein, list_tag in data:
            if not pathlib.Path(f"{params['protvec_encoding_dir']}/{name_pro}.pt").is_file() and \
               not pathlib.Path(f"{params['protvec_encoding_dir_tmp']}/{name_pro}.pt").is_file():
                missing_protvec.append((name_pro, protein, list_tag))
        
        # Process missing proteins
        for name_pro, protein, list_tag in missing_protvec:
            compute_protvec_embeddings(name_pro, protein, params["protvec_dict"], params["protvec_encoding_dir_tmp"])
    
    # Process ESM-2 embeddings as before
    missing_esm2 = []
    for name_pro, protein, list_tag in data:
        if params["initialization"] == "ESM-2":
            if not pathlib.Path(f"{ESM_2_PATH}/{name_pro}.pt").is_file():
                missing_esm2.append((name_pro, protein, list_tag))
    if missing_esm2:
        batch_call_esm2(missing_esm2, params)
        
    # Process each protein
    for name_pro, protein, list_tag in data:
        if params["initialization"] == "ESM-2":
            list_tag = list_tag[:min(len(list_tag), 1023)]
        length_pro = len(list_tag)
        
        # Handle ProtVec embeddings
        if params["initialization"] == "ProtVec":
            # Try to load from permanent directory first, then temp directory
            protvec_file = None
            if pathlib.Path(f"{params['protvec_encoding_dir']}/{name_pro}.pt").is_file():
                protvec_file = f"{params['protvec_encoding_dir']}/{name_pro}.pt"
            elif pathlib.Path(f"{params['protvec_encoding_dir_tmp']}/{name_pro}.pt").is_file():
                protvec_file = f"{params['protvec_encoding_dir_tmp']}/{name_pro}.pt"
            
            if protvec_file:
                x = torch.load(protvec_file)
                # For BiLSTM, we need to reshape the embedding
                if params["model"] == "BiLSTM":
                    # Repeat the embedding for each amino acid in the sequence
                    x_expanded = x.unsqueeze(0).repeat(length_pro, 1)
                    padded_peps = torch.zeros((1, length_pro, 100))
                    padded_peps[0, :, :] = x_expanded
                else:
                    padded_peps = x
            else:
                print(f"Error: ProtVec embedding for {name_pro} not found")
                continue
        elif params["initialization"] == "ESM-2":
            # Intentamos cargar el embedding ya preprocesado
            try:
                loaded = torch.load(f"{ESM_2_PATH}/{name_pro}.pt")
                # Se espera que el dicionario tenga la representación en key "representations"[33]
                if isinstance(loaded, dict):
                    x = loaded["representations"][33]
                else:
                    x = loaded
                # Actualizamos el guardado por si no está en el formato esperado
                torch.save(x, f"{ESM_2_PATH}/{name_pro}.pt")
            except Exception as e:
                # Si falla, se salta la proteína
                print(f"Error al cargar ESM-2 para {name_pro}: {e}")
                continue
            if params["model"] == "BiLSTM":
                # Se limita el tamaño en función del máximo
                padded_peps = torch.zeros((1, min(length_pro, 1023), 1280))
                padded_peps[0, :x.shape[0], :] = x
            else:
                padded_peps = x  # o se asigna de forma directa según el modelo
        else:
            # Caso en que se use otra inicialización (no se trata aquí)
            x = convert_pep_to_list_of_aa(protein, amino_to_idx)
            x = torch.LongTensor(x).squeeze()
            padded_peps = torch.zeros((1, length_pro), dtype=torch.long)
            padded_peps[0, :] = x

        if params["model"] == "GCN":
            list_edges = []
            path_dist = f"{DIST_PATH}/{name_pro}.csv" if pathlib.Path(f"{DIST_PATH}/{name_pro}.csv").is_file() \
                else f"{params['dist_dir']}/{name_pro}.csv"
            with open(path_dist) as dist_file:
                for line in dist_file:
                    line = line.strip().split(',')
                    if float(line[2]) <= params["dist"]:
                        if not line[0][-1].isdigit():
                            line[0] = line[0][:-1]
                        if not line[1][-1].isdigit():
                            line[1] = line[1][:-1]
                        list_edges.append([int(line[0]) - 1, int(line[1]) - 1, float(line[2])])
            list_aa = protein
            rows = [e[0] for e in list_edges]
            cols = [e[1] for e in list_edges]
            edge_index = SparseTensor(
                row=torch.tensor(rows, dtype=torch.long), 
                col=torch.tensor(cols, dtype=torch.long),
                sparse_sizes=(len(list_aa), len(list_aa))
            )
            data_obj = Data(x=x, edge_index=edge_index, y=torch.tensor(list_tag, dtype=torch.float))
            batches_list.append([data_obj, protein, name_pro])
        else:
            batch_size = 1
            bio_tensor = []
            if params["initialization"] == "Kidera+bio":
                for j in range(len(protein)):
                    aa_prop = calc_properties_of_amino_acids(protein[max(0, j - 4):min(len(protein), j + 5)])
                    for i, val in enumerate(aa_prop):
                        dict_bio_prop[i].append(val)
                    bio_tensor.append(aa_prop)
            # Para otros casos (incluyendo ESM-2), se usa padded_peps ya definido
            batch_signs = torch.FloatTensor(list_tag)
            dict_batch = {"embeding_pro": padded_peps,
                          "aa_prop": bio_tensor,
                          "signs": batch_signs,
                          "protein": protein,
                          "name_pro": name_pro
                         }
            batches_list.append(dict_batch)

    # Normalización en el caso Kidera+bio, si aplica
    if params["initialization"] == "Kidera+bio":
        if train:
            list_aa_prop = []
            for i, list_vals in dict_bio_prop.items():
                list_aa_prop.append([np.mean(list_vals), np.std(list_vals)])
        for batch in batches_list:
            bio_lists = batch["aa_prop"]
            bio_tensor = torch.zeros((1, len(bio_lists), 8))
            for i, i_list in enumerate(bio_lists):
                for j, (meav_j, std_j) in enumerate(list_aa_prop):
                    i_list[j] = (i_list[j] - meav_j) / std_j
                bio_tensor[0, i, :] = torch.FloatTensor(i_list)
            batch["aa_prop"] = bio_tensor
    else:
        list_aa_prop = []
        
    print(len(batches_list))
    return batches_list, list_aa_prop

def create_batches_rsa(data, params, amino_to_idx, list_aa_prop=None, train=False):
    batches_list = []
    dict_bio_prop = {}
    for i in range(8):
        dict_bio_prop[i] = []
    for name_pro, protein, list_tag in data:
        # print(f"{name_pro}")
        # print(f"{name_pro},{len(protein)},{len(list_tag)}")
        list_rsa = []
        if not pathlib.Path(f"netsurfp_rsa/{name_pro}.csv").is_file():
            continue
        with open(f"netsurfp_rsa/{name_pro}.csv", mode='r') as infile:
            reader = csv.DictReader(infile, delimiter=',')
            for row in reader:
                list_rsa.append(float(row[" rsa"]))

        if params["initialization"] == "ESM-2":
            list_tag = list_tag[:min(len(list_tag), 1023)]
            list_rsa = list_rsa[:min(len(list_rsa), 1023)]
        length_pro = len(list_tag)
        
        # Handle ProtVec initialization
        if params["initialization"] == "ProtVec":
            # Try to load from permanent directory first, then temp directory
            protvec_file = None
            if pathlib.Path(f"{params['protvec_encoding_dir']}/{name_pro}.pt").is_file():
                protvec_file = f"{params['protvec_encoding_dir']}/{name_pro}.pt"
            elif pathlib.Path(f"{params['protvec_encoding_dir_tmp']}/{name_pro}.pt").is_file():
                protvec_file = f"{params['protvec_encoding_dir_tmp']}/{name_pro}.pt"
            
            if protvec_file:
                x = torch.load(protvec_file)
                # Add RSA values to the embedding
                if len(list_rsa) != length_pro:
                    continue
                
                # For BiLSTM, we need to reshape the embedding and add RSA
                if params["model"] == "BiLSTM":
                    # Repeat the embedding for each amino acid in the sequence
                    x_expanded = x.unsqueeze(0).repeat(length_pro, 1)
                    # Concatenate RSA values
                    x_with_rsa = torch.cat((x_expanded, torch.FloatTensor(list_rsa).unsqueeze(1)), dim=1)
                    padded_peps = torch.zeros((1, length_pro, 101))  # 100 + 1 for RSA
                    padded_peps[0, :, :] = x_with_rsa
                else:
                    # For other models, handle differently if needed
                    padded_peps = torch.cat((x, torch.FloatTensor(list_rsa).unsqueeze(0)), dim=0)
            else:
                print(f"Error: ProtVec embedding for {name_pro} not found")
                continue
        elif params["initialization"] == "ESM-IF1":
            if pathlib.Path(f"{ESM_IF1_PATH}/{name_pro}.pt").is_file():
                x = torch.load(f"{ESM_IF1_PATH}/{name_pro}.pt")
            elif pathlib.Path(params["esmif1_encoding_dir"]).is_file():
                x = torch.load(params["esmif1_encoding_dir"])
                if not x.shape[0] == length_pro:
                    x = call_esmif1(name_pro, params["esmif1_encoding_dir"], params["pdb_dir"])
            else:
                try:
                    x = call_esmif1(name_pro, params["esmif1_encoding_dir"], params["pdb_dir"])
                except:
                    print(f"######invalid esmif1 {name_pro}")
                    continue

            if not x.shape[0] == length_pro:
                continue
            if not x.shape[0] == len(list_rsa):
                continue
            x = torch.cat((x, torch.unsqueeze(torch.FloatTensor(list_rsa), 1)), dim=1)
            if params["model"] == "BiLSTM":
                padded_peps = torch.zeros((1, min(length_pro, 1023), 513))
                padded_peps[0, :, :] = x

        elif params["initialization"] == "ESM-2":
            if pathlib.Path(f"{ESM_2_PATH}/{name_pro}.pt").is_file():
                try:
                    x = torch.load(f"{ESM_2_PATH}/{name_pro}.pt")["representations"][33]
                    torch.save(
                        x,
                        f"{ESM_2_PATH}/{name_pro}.pt",
                    )
                except:
                    try:
                        x = torch.load(f"{ESM_2_PATH}/{name_pro}.pt")
                        # print("x", x.shape)
                        if not x.shape[0] == length_pro:
                            x = call_esm2(name_pro, protein, params["esm2_encoding_dir"])
                    except:
                        x = call_esm2(name_pro, protein, params["esm2_encoding_dir"])
            elif pathlib.Path(f"params['esm2_encoding_dir']/{name_pro}.pt").is_file():
                try:
                    x = torch.load(params["esm2_encoding_dir"])
                except:
                    x = call_esm2(name_pro, protein, params["esm2_encoding_dir"])
            else:
                x = call_esm2(name_pro, protein, params["esm2_encoding_dir"])

            if not x.shape[0] == len(list_rsa):
                continue
            x = torch.cat((x, torch.unsqueeze(torch.FloatTensor(list_rsa), 1)), dim=1)
            if params["model"] == "BiLSTM":
                padded_peps = torch.zeros((1, min(length_pro, 1023), 1281))
                print("padded_peps", x.shape)
                padded_peps[0, :, :] = x

        else:
            x = convert_pep_to_list_of_aa(protein, amino_to_idx)
            x = torch.LongTensor(x)
            x = torch.squeeze(x)

        if params["model"] == "GCN":
            list_edges = []
            path_dist = f"{DIST_PATH}/{name_pro}.csv" if pathlib.Path(f"{DIST_PATH}/{name_pro}.csv").is_file() \
                else f"{params['dist_dir']}/{name_pro}.csv"
            with open(path_dist) as dist_file:
                for line in dist_file:
                    line = line.strip().split(',')
                    if float(line[2]) <= params["dist"]:
                        if not line[0][-1].isdigit():
                            line[0] = line[0][:-1]
                        if not line[1][-1].isdigit():
                            line[1] = line[1][:-1]
                        list_edges.append([int(line[0]) - 1, int(line[1]) - 1, float(line[2])])
            list_aa = protein  # convert_pep_to_list_of_aa(protein, amino_to_idx)

            rows = [e[0] for e in list_edges]
            cols = [e[1] for e in list_edges]
            # Create sparse tensor
            edge_index = SparseTensor(
                row=torch.tensor(rows, dtype=torch.long), col=torch.tensor(cols, dtype=torch.long),

                sparse_sizes=(len(list_aa), len(list_aa))
            )  # value=torch.tensor(vals, dtype=torch.float),
            data = Data(x=x, edge_index=edge_index, y=torch.tensor(list_tag, dtype=torch.float))
            batches_list.append([data, protein, name_pro])
        else:
            batch_size = 1

            bio_tensor = []
            if params["initialization"] == "Kidera+bio":
                # torch.zeros((batch_size, length_pep, padding))
                for j in range(0, len(protein)):
                    aa_prop = calc_properties_of_amino_acids(protein[max(0, j - 4):min(len(protein), j + 5)])
                    for i, val in enumerate(aa_prop):
                        dict_bio_prop[i].append(val)
                    bio_tensor.append(aa_prop)
                    # bio_tensor[0, j, :] = torch.FloatTensor(aa_prop)

            if params["initialization"] == "ESM-2":
                padded_peps = torch.zeros((batch_size, min(length_pro, 1023), 1281))
                padded_peps[0, :, :] = x
            elif params["initialization"] == "ESM-IF1":
                padded_peps = torch.zeros((batch_size, length_pro, params["embedding_dim"]))
                padded_peps[0, :, :] = x
            else:
                padded_peps = torch.zeros((batch_size, length_pro), dtype=torch.long)
                padded_peps[0, :] = x

            batch_signs = torch.torch.FloatTensor(list_tag)
            dict_batch = {"embeding_pro": padded_peps,
                          "aa_prop": bio_tensor,
                          "signs": batch_signs,
                          "protein": protein,
                          "name_pro": name_pro
                          }

            batches_list.append(dict_batch)

    if params["initialization"] == "Kidera+bio":
        if train:
            list_aa_prop = []
            for i, list_vals in dict_bio_prop.items():
                std = np.std(list_vals)
                mean = np.mean(list_vals)
                list_aa_prop.append([mean, std])

        for batch in batches_list:
            bio_lists = batch["aa_prop"]
            bio_tensor = torch.zeros((1, len(bio_lists), 8))
            for i, i_list in enumerate(bio_lists):
                for j, (meav_j, std_j) in enumerate(list_aa_prop):
                    i_list[j] = (i_list[j] - meav_j) / std_j
                bio_tensor[0, i, :] = torch.FloatTensor(i_list)
            batch["aa_prop"] = bio_tensor
    else:
        list_aa_prop = []

    print(len(batches_list))
    return batches_list, list_aa_prop



    batches_list = []
    dict_bio_prop = {}
    for i in range(8):
        dict_bio_prop[i] = []
    for name_pro, protein, list_tag in data:
        if params["initialization"] == "ESM-2":
            list_tag = list_tag[:min(len(list_tag), 1023)]

        length_pro = len(list_tag)

        if params["initialization"] == "ESM-IF1":
            if pathlib.Path(f"{ESM_IF1_PATH}/{name_pro}.pt").is_file():
                x = torch.load(f"{ESM_IF1_PATH}/{name_pro}.pt")
            elif pathlib.Path(params["esmif1_encoding_dir"]).is_file():
                x = torch.load(params["esmif1_encoding_dir"])
                print(f"x.shape: {x.shape}")
                if not x.shape[0] == length_pro:
                    x = call_esmif1(name_pro, params["esmif1_encoding_dir"], params["pdb_dir"])
            else:
                try:
                    x = call_esmif1(name_pro, params["esmif1_encoding_dir"], params["pdb_dir"])
                except:
                    print(f"######invalid esmif1 {name_pro}")
                    continue

            if not x.shape[0] == length_pro:
                continue

            if params["model"] == "BiLSTM":
                padded_peps = torch.zeros((1, length_pro, 512))
                padded_peps[0, :, :] = x

        elif params["initialization"] == "ESM-2":
            if pathlib.Path(f"{ESM_2_PATH}/{name_pro}.pt").is_file():
                try:
                    x = torch.load(f"{ESM_2_PATH}/{name_pro}.pt")["representations"][33]
                    torch.save(
                        x,
                        f"{ESM_2_PATH}/{name_pro}.pt",
                    )
                except:
                    try:
                        x = torch.load(f"{ESM_2_PATH}/{name_pro}.pt")
                        if not x.shape[0] == length_pro:
                            x = call_esm2(name_pro, protein, params["esm2_encoding_dir"])
                    except:
                        x = call_esm2(name_pro, protein, params["esm2_encoding_dir"])
            elif pathlib.Path(f"params['esm2_encoding_dir']/{name_pro}.pt").is_file():
                try:
                    x = torch.load(params["esm2_encoding_dir"])
                except:
                    x = call_esm2(name_pro, protein, params["esm2_encoding_dir"])
            else:
                x = call_esm2(name_pro, protein, params["esm2_encoding_dir"])

            if params["model"] == "BiLSTM":
                padded_peps = torch.zeros((1, min(length_pro, 1023), 1280))
                padded_peps[0, :, :] = x

        else:
            x = convert_pep_to_list_of_aa(protein, amino_to_idx)
            x = torch.LongTensor(x)
            x = torch.squeeze(x)

        if params["model"] == "GCN":
            list_edges = []
            path_dist = f"{DIST_PATH}/{name_pro}.csv" if pathlib.Path(f"{DIST_PATH}/{name_pro}.csv").is_file() \
                else f"{params['dist_dir']}/{name_pro}.csv"
            with open(path_dist) as dist_file:
                for line in dist_file:
                    line = line.strip().split(',')
                    if float(line[2]) <= params["dist"]:
                        if not line[0][-1].isdigit():
                            line[0] = line[0][:-1]
                        if not line[1][-1].isdigit():
                            line[1] = line[1][:-1]
                        list_edges.append([int(line[0]) - 1, int(line[1]) - 1, float(line[2])])
            list_aa = protein  # convert_pep_to_list_of_aa(protein, amino_to_idx)

            rows = [e[0] for e in list_edges]
            cols = [e[1] for e in list_edges]
            # Create sparse tensor
            edge_index = SparseTensor(
                row=torch.tensor(rows, dtype=torch.long), col=torch.tensor(cols, dtype=torch.long),

                sparse_sizes=(len(list_aa), len(list_aa))
            )  # value=torch.tensor(vals, dtype=torch.float),
            data = Data(x=x, edge_index=edge_index, y=torch.tensor(list_tag, dtype=torch.float))
            batches_list.append([data, protein, name_pro])
        else:
            batch_size = 1

            bio_tensor = []
            if params["initialization"] == "Kidera+bio":
                # torch.zeros((batch_size, length_pep, padding))
                for j in range(0, len(protein)):
                    aa_prop = calc_properties_of_amino_acids(protein[max(0, j - 4):min(len(protein), j + 5)])
                    for i, val in enumerate(aa_prop):
                        dict_bio_prop[i].append(val)
                    bio_tensor.append(aa_prop)
                    # bio_tensor[0, j, :] = torch.FloatTensor(aa_prop)

            if params["initialization"] == "ESM-2":
                padded_peps = torch.zeros((batch_size, min(length_pro, 1023), 1280))
                padded_peps[0, :, :] = x
            elif params["initialization"] == "ESM-IF1":
                padded_peps = torch.zeros((batch_size, length_pro, params["embedding_dim"]))
                padded_peps[0, :, :] = x
            else:
                padded_peps = torch.zeros((batch_size, length_pro), dtype=torch.long)
                padded_peps[0, :] = x

            batch_signs = torch.torch.FloatTensor(list_tag)
            dict_batch = {"embeding_pro": padded_peps,
                          "aa_prop": bio_tensor,
                          "signs": batch_signs,
                          "protein": protein,
                          "name_pro": name_pro
                          }

            batches_list.append(dict_batch)

    if params["initialization"] == "Kidera+bio":
        if train:
            list_aa_prop = []
            for i, list_vals in dict_bio_prop.items():
                std = np.std(list_vals)
                mean = np.mean(list_vals)
                list_aa_prop.append([mean, std])

        for batch in batches_list:
            bio_lists = batch["aa_prop"]
            bio_tensor = torch.zeros((1, len(bio_lists), 8))
            for i, i_list in enumerate(bio_lists):
                for j, (meav_j, std_j) in enumerate(list_aa_prop):
                    i_list[j] = (i_list[j] - meav_j) / std_j
                bio_tensor[0, i, :] = torch.FloatFloatTensor(i_list)
            batch["aa_prop"] = bio_tensor
    else:
        list_aa_prop = []

    print(len(batches_list))
    return batches_list, list_aa_prop