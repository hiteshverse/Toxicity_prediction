from utils import utils
from Evaluation import Evaluation
import pandas as pd
import numpy as np
#------------------------------------------------------------------------------------#


#that code is use to create dataframe for prdicted toxicity
def ALL_MODEL_PREDICTION(smi_list,dict):
    '''
    Plase provide all information in smile list and dictionary formate
    input-
    smiles:they you want to calculate toxicity,
    dict: contain the dataset about key:values pair of name and model about toxictiy
    '''
    data=[]
    data.append(smi_list)
    for k,l in zip(dict.keys(),dict.values()):
        if str(k)=='Hepatotoxicity':
            rdkit=Evaluation.HEPATO_PREDICTION(smi_list,l)
            data.append(rdkit)
            continue
        elif str(k)=='Cardiotoxicity':
            rdkit=Evaluation.CARDIO_PREDICTION(smi_list,l)
            data.append(rdkit)
            continue
        elif str(k)=='Carcinogenicity':
            rdkit=Evaluation.CARCINO_PREDICTION(smi_list,l)
            data.append(rdkit)
            continue
        else: 
            if str(k) in [str(i) for i in dict.keys() if str(i) not in ['Hepatotoxicity','Cardiotoxicity']]:
                ecfp=Evaluation.MUTA_NEPHRO_NURO_PREDICTION(smi_list,l)
                data.append(ecfp)
            
    data_final=pd.DataFrame(np.array(data).T,columns=['SMILES']+list(dict.keys()))
    return data_final 


#------------------------------------------------------------------------------------#

if __name__=='__main__':
       
    ## this is the example of the prdict that file in main.py
    smiles=['CC(C)(C[n+]1ccccc1)c1cc(F)c2[nH]c(CC=O)cc2c1',
            'CCCCn1cnc2c(c1=O)N(CC(=O)OCC)C=C(C#N)S2',
            'Cc1cccc(NCC(O)COc2c(C)cc(C)nc2C)c1',
            'O=P(O)(O)c1ccc(NC(=S)Cc2ccc(F)cc2)cc1',
            'NC(=O)c1nsc(C(=O)Nc2cccc3ccccc23)c1N',
            'CC(O)(Cl)Oc1cc(CCCN)c(F)cc1O']
    
    tox_name=['Hepatotoxicity','Mutagenicity','Cardiotoxicity','Nephrotoxicity']

    ## create key:values pair of pickle model with its name eg.'hepatotoxicity':ExtraTreesClassifier()
    new_dict=utils.model_para('models',tox_name)

    ## create and print the final dataset files
    df=ALL_MODEL_PREDICTION(smiles,new_dict)
    print(df)

    #print(model1)
