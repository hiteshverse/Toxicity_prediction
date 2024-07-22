import pickle
import glob
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from rdkit.Chem import MACCSkeys
from rdkit.ML.Descriptors import MoleculeDescriptors
import itertools
#--------------------------------------------------------------------------#

class utils:

    def RDKIT_DESCRIPTORS(smile):
        # calculate the descriptores
        calc = MoleculeDescriptors.MolecularDescriptorCalculator([x[0] for x in Descriptors._descList])
        header=calc.GetDescriptorNames()
        descriptors=[]
        mol=Chem.MolFromSmiles(smile[0])
        if mol is None:
            descriptors.append(tuple(itertools.repeat(np.nan, 209)))
        else:
            ds=calc.CalcDescriptors(mol)
            descriptors.append(ds) 

        desc=pd.DataFrame(descriptors,columns=header)
        return desc  
    #-----------------------------------------------------------------------------#
     
    def ECFP4(smiles):
        res=[]
        try:
            mol = Chem.MolFromSmiles(smiles[0])
            ecfp4_fingerprint = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=1024)
            res.append(list(ecfp4_fingerprint))
        except:
            res.append(list(itertools.repeat(np.nan, 1024)))

        df = pd.DataFrame(res, columns=[f'B{i}' for i in range(1024)])
        return df
    
    #----------------------------------------------------------------------------------#
    # def SUMMARY_TABLE(df):
    #     print(df)
    #     data=df.copy()


    #     def replace_values(x):
    #         if isinstance(x, float):  # Check if the value is a float
    #             if x <= 0.5:
    #                 return '1'
    #             else:
    #                 return '0'
    #         else:
    #             return x  # Return string values as they are

    #     # Apply the function to each column
    #     for i in df.columns:
    #         df[i]=df[i].map(replace_values)
    #         print(i)
    #     print(df)
    #     data1=data.applymap(replace_values)
    #     print(data1)
    #     s = pd.DataFrame()
    #     for i in data1.columns[1:]:
    #         s1 = data1[str(i)].value_counts().to_dict()
    #         s = pd.concat([s,pd.DataFrame([s1],index=[i])])

    #     s.fillna(0,inplace=True)
    #     fin = s.T
    #     fin = fin.astype(int)
    #     fin.columns=data1.columns[1:]
    #     #q=[ 'Toxic' if i=='1' else 'Non-Toxic' for i in fin.index]
    #     q = ['Not_calculate' if i == 'Not_calculate' else 'Toxic' if i =='1' else 'Non-Toxic' for i in fin.index]
    #     fin.index=q
    #     fin = fin.reset_index(names='Class')
    #     # print(fin)
    #     return fin


    def SUMMARY_TABLE(df):
        data = df.copy()

        # Convert float values to binary using pd.cut
        for column in data.columns[1:]:
            data[column] = data[column].apply(lambda x: '1' if isinstance(x, (float, int)) and x <= 0.5 else ('0' if isinstance(x, (float, int)) else x))
        
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
    #---------------------------------------------------------------------------------#
    ##exract model and data for pickle
    def LOAD_MODEL(model_folder,tox_name):
        models = {}
        paths = glob.glob(f'{model_folder}/*')
        
        for i in paths:
            models[i.split('/')[1][:-4]] =  pickle.load(open(i, 'rb'))  ##extract the all names and models for the dataset 

        tox_dict = {name: models[name] for name in tox_name if name in models}
        return tox_dict