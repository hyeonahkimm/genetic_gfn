import os
from tdc import Oracle
from rdkit import Chem

from genetic_gfn.scoring_function import docking, reverse_sigmoid_transformation


if __name__ == '__main__':
    # scores = [(-8.76, 0.795, 2.994), (-9.11, 0.726, 2.823), (-10.83, 0.380, 3.638), (-10.75, 0.392, 2.649), (-11.02, 0.386, 2.550), (-12.85, 0.641, 3.231)]
    scores = [(-8.33, 0.719, 2.959), (-8.89, 0.656, 2.854), (-11.26, 0.262, 3.520), (-11.30, 0.275, 2.917), (-11.84, 0.278, 2.894), (-13.76, 0.356, 3.566)]
    # scores = [(-11.3, 0.310, 2.530), (-11.1, 0.258, 2.729), (-11.2, 0.214, 2.549), (-12.3, 0.237, 2.772), (-13.10, 0.253, 3.104), (-13.2, 0.241, 2.806)]
    # scores = [(-8.33, 0.719, 2.959), (-8.89, 0.656, 2.854), (-11.26, 0.262, 3.520), (-11.30, 0.275, 2.917), (-11.84, 0.278, 2.894)]

    for d, qed, sa in scores:
        print(0.8 * reverse_sigmoid_transformation(d) + 0.1 * qed + 0.1 * (10 - sa) / 9)
    # receptor = '7jir+w2'
    # box_center = [51.51, 32.16, -0.55]
    # smiles = ['O=C(c1cccc(-c2ccc3c(-c4nc5cc(F)c(F)cc5[nH]4)n[nH]c3c2)c1)c1ccccc1',
    #           'c1(-c2cc(-c3cc4c(cc3)snn4)ccc2)ccc(C(NCc2cc(C(=O)Nc3nn[nH]n3)ccc2F)=O)cc1',
    #           'c1c(-c2c(C)ccc(C(=O)c3cc4ccccc4[nH]3)c2)nc(-c2ccnc(-c3cccc(-c4ccccc4)c3)c2)cc1',
    # 'O=c1[nH]c(-c2c[nH]c3cccc(F)c23)nc(-c2cccc3cccc(F)c23)n1',
    # 'O=c1nc(-c2cccc3occc23)cc(-c2c[nH]c3ccc(F)cc23)[nH]1',]

    receptor = 'RDRP'
    box_center = [93.88, 83.08, 97.29]
    smiles = ['O=C1CCN(c2nc(-c3cccc(C(=O)N4CCN(c5cc(-c6ccccc6)nc6ccccc56)CC4)c3)cc3ccccc23)CCN1',
              'O=C1CCN(c2nc(-c3cccc(C(=O)N4CCN(c5cc(-c6nc7ccccc7c(=O)[nH]6)c6ccccc6n5)CC4)c3)nc3ccccc23)CCN1']
    
    oracle_QED = Oracle(name='QED')
    oracle_SA = Oracle(name='SA')

    for smi in smiles:
        mol = Chem.MolFromSmiles(smi)
        print(mol.GetNumAtoms())
        if mol == None:
            continue
        docking_score, unnormalized = docking(smi, receptor_file="data/targets/{}.pdbqt".format(receptor), box_center=box_center, return_raw=True)
        qed = oracle_QED(smi)
        sa = oracle_SA(smi)  #(10 - oracle_SA(smiles[i])) / 9
        scores = [0.8 * docking_score + 0.1 * qed + 0.1 * (10 - sa) / 9, unnormalized, qed, sa]
        print(smi)
        print(f'score: {scores[0]}, docking: {scores[1]}, qed: {scores[2]}, sa: {scores[3]}')
        try:
            ligand_mol_file = f"./docking/tmp/mol_{smi}.mol"
            ligand_pdbqt_file = f"./docking/tmp/mol_{smi}.pdbqt"
            docking_pdbqt_file = f"./docking/tmp/dock_{smi}.pdbqt"
            for filename in [ligand_mol_file, ligand_pdbqt_file, docking_pdbqt_file]:
                if os.path.exists(filename):
                    os.remove(filename)
        except:
            pass