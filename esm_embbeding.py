import sys
import subprocess
import torch

ESM_SCRIPT_PATH =  "esm_utils/extract.py"

class esm_embbeding():
    def __init__(self, accs, seqs,  esm_encoding_dir):
        self.accs = accs
        self.seqs = seqs
        self.esm_encoding_dir = esm_encoding_dir

    def create_fasta_for_esm_transformer(self):
        """
        Outputs a FASTA file with accession and sequences format,
        that can be read by the ESM-2 transformer.
        """
        uppercase_entries = list()
        # Convert todas las secuencias a mayúsculas
        entries = list(zip(self.accs, self.seqs))
        for acc, sequence in entries:
            uppercase_entries.append((acc, sequence.upper()))
            
        # Usamos un nombre de archivo fijo (o podrías generar uno dinámico)
        fasta_file = f"{self.esm_encoding_dir}/antigens_batch.fasta"
        with open(fasta_file, "w") as outfile:
            output = ""
            for acc, sequence in uppercase_entries:
                output += f">{acc}\n{sequence}\n"
            outfile.write(output.strip())

    def call_esm_script(self):
        fastaPath = f"{self.esm_encoding_dir}/antigens_batch.fasta"
        try:
            subprocess.check_call(
                ['python', ESM_SCRIPT_PATH, "esm2_t33_650M_UR50D", fastaPath, self.esm_encoding_dir, "--include", "per_tok"]
            )
        except subprocess.CalledProcessError as error:
            sys.exit(
                f"ESM model could not be run with error: {error}.\nThis is likely a memory issue."
            )

    def prepare_esm_data(self):
        esm_representations = list()
        for acc in self.accs:
            esm_encoded_acc = torch.load(f"{self.esm_encoding_dir}/{acc}.pt")
            esm_representation = esm_encoded_acc["representations"][33]
            esm_representations.append(esm_representation)
        return esm_representations