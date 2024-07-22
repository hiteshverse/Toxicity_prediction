import pickle
import glob
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
import itertools

class utils:

    @staticmethod
    def RDKIT_DESCRIPTORS(smile):
        # calculate the descriptors
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
        header = calc.GetDescriptorNames()
        descriptors = []
        mol = Chem.MolFromSmiles(smile[0])
        if mol is None:
            descriptors.append(tuple(itertools.repeat(np.nan, 209)))
        else:
            ds = calc.CalcDescriptors(mol)
            descriptors.append(ds)

        desc = pd.DataFrame(descriptors, columns=header)
        return desc

    @staticmethod
    def ECFP4(smiles):
        res = []
        try:
            mol = Chem.MolFromSmiles(smiles[0])
            ecfp4_fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            res.append(list(ecfp4_fingerprint))
        except:
            res.append(list(itertools.repeat(np.nan, 1024)))

        df = pd.DataFrame(res, columns=[f'B{i}' for i in range(1024)])
        return df

    # @staticmethod
    def SUMMARY_TABLE(df):
        df.replace('Not_calculate', np.nan, inplace=True)
        data1 = df.copy()
        data2=data1.iloc[:,1:]
        data=data2.astype(float)

        # Convert float values to binary using pd.cut
        for column in data.columns[1:]:
            data[column] = data[column].apply(lambda x: 'Not_calculate' if pd.isna(x) else '1' if x <= 0.5 else '0')

        print(data)
        # Creating summary table
        s = pd.DataFrame()
        for column in data.columns[1:]:
            s1 = data[column].value_counts().to_dict()
            s = pd.concat([s, pd.DataFrame([s1], index=[column])])
        
        s.fillna(0, inplace=True)
        fin = s.T
        fin = fin.astype(int, errors='ignore')
        fin.columns = data.columns[1:]

        # Creating index labels
        fin.index = ['Not_calculate' if i == 'Not_calculate' else 'Toxic' if i == '1' else 'Non-Toxic' for i in fin.index]
        fin = fin.reset_index(names='Class')
        
        return fin
    
    

    @staticmethod
    def LOAD_MODEL(model_folder, tox_name):
        models = {}
        paths = glob.glob(f'{model_folder}/*')
        
        for i in paths:
            model_name = i.split('/')[-1][:-4]  # Corrected the index to get the model name correctly
            models[model_name] = pickle.load(open(i, 'rb'))  # extract the all names and models for the dataset 

        tox_dict = {name: models[name] for name in tox_name if name in models}
        return tox_dict

# Example usage:
if __name__ == "__main__":
    data = {
        'SMILES': [
            'CS(=O)(=O)c1ccc(Cn2nc3cccc(-c4ccc(Cc5nc6ccc(C(...',
            'Fc1ccc(CN2N=C3C=CC=C(N3C2=O)c2ccc(cc2)Cc2nc3cc...',
            'N#Cc1ccc(CN2N=C3C=CC=C(N3C2=O)c2ccc(cc2)Cc2nc3...',
            'Fc1ccc(cc1)CN1N=C2C=CC=C(N2C1=O)c1ccc(cc1)Cc1n...'
        ],
        'Hepatotoxicity': [0.76, 0.76, 0.74, 'ab'],
        'Mutagenicity': [0.38, 0.32, 0.39, 0.35],
        'Carcinogenicity': [0, 0, 0, 0],
        'Cardiotoxicity': [0.65, 0.94, 0.96, 0.93],
        'Nephrotoxicity': [0.5, 0.49, 0.49, 0.53]
    }

    df = pd.DataFrame(data)
    summary_table = utils.SUMMARY_TABLE(df)
    print(summary_table)
