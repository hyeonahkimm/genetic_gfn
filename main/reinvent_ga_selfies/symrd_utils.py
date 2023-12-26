import torch
import selfies as sf
from rdkit import Chem

def seq_to_selfies(seqs, voc):
    selfies_list = []
    for seq in seqs.cpu().numpy():
        selfies_list.append(voc.decode_selfies(seq))
    return selfies_list 


def selfies_to_seq(selfies, voc):
    tokenized = voc.tokenize(selfies)
    seq = []
    for char in tokenized:
        seq.append(voc.vocab[char])
    return torch.tensor(seq).float().cuda()


def make_symmetric_selfies(seqs, voc):
    padded_len = seqs.shape[1]
    selfies_list = seq_to_selfies(seqs, voc)

    symmetric_selfies_list = []
    symmetric_seq_list = []
    for i, selfies in enumerate(selfies_list):
        smile = sf.decoder(selfies)

        mol = Chem.MolFromSmiles(smile)
        canonical_smiles = Chem.MolToSmiles(mol, isomericSmiles=False)
        
        try:
            new_selfies = sf.encoder(canonical_smiles)
            tokenized = voc.tokenize(new_selfies)

            new_seq = selfies_to_seq(new_selfies, voc)
            new_seq = torch.cat([new_seq, torch.zeros(padded_len-len(new_seq)).cuda()], dim=0)

        except:
            new_seq = seqs[i]
            new_selfies = selfies

        symmetric_selfies_list.append(new_selfies)
        symmetric_seq_list.append(new_seq)

    return torch.stack(symmetric_seq_list)
