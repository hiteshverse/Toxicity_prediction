import numpy as np
from utils import utils
import pickle
#-----------------------------------------------------------------------------#

## Missing value imputation - Hepatotoxicity
model_MRLOW=pickle.load(open('models/hepeto_BCUT2D_MRLOW.pkl', 'rb'))  
model_CHGLO=pickle.load(open('models/hepeto_BCUT2D_CHGLO.pkl', 'rb'))

## Missing value imputation - Carditotoxicity
cardio_KNN_IMPUTER=pickle.load(open('models/Cardio_knn_impute.pkl', 'rb'))  
cardio_MINMAX_SCALER=pickle.load(open('models/Cardio_minmax_scaler.pkl', 'rb'))

#-----------------------------------------------------------------------------#
class Evaluation:
    def __init__(self,smi,model):
        self.smi=smi
        self.model=model
    
    def HEPATO_PREDICTION(smi,model):
        pred=[]
        for i in smi:
            x=utils.RDKIT_DESCRIPTORS([i])

           #Extract relavant feature
            x.drop(columns=['Ipc','AvgIpc','MinAbsPartialCharge', 'MaxAbsPartialCharge', 'MinPartialCharge','MaxPartialCharge', 'BCUT2D_CHGHI', 'BCUT2D_MRHI', 'BCUT2D_LOGPLOW',
                            'BCUT2D_LOGPHI', 'BCUT2D_MWHI', 'BCUT2D_MWLOW','NumRadicalElectrons','SMR_VSA8','SlogP_VSA9','fr_isocyan','fr_prisulfonamd','fr_thiocyan'],inplace=True)   #'AvgIpc'
            
            ### Extract relavant for handling missing values
            para_mrlow=['MaxEStateIndex','SMR_VSA1','FractionCSP3','NumAliphaticRings','NumSaturatedRings','fr_quatN','Chi4n','Chi3n','NumSaturatedHeterocycles','qed',
                        'MinEStateIndex','MinAbsEStateIndex','BalabanJ']
            para_chglo=['NumAliphaticRings','Chi4n','NumAliphaticCarbocycles','RingCount', 'SlogP_VSA4','fr_bicyclic','BalabanJ','MinAbsEStateIndex','FpDensityMorgan1',
                        'qed','NumAromaticRings','NumAromaticHeterocycles','fr_Ar_N']
            
            if x.isnull().sum().sum()==0:
                pred.append(np.round(model.predict_proba(np.asarray(x))[0][1],2))
            elif x.isnull().sum().sum()==2:
                pred_low=model_MRLOW.predict(x[para_mrlow])
                pred_lo=model_CHGLO.predict(x[para_chglo])
                x['BCUT2D_CHGLO']=pred_lo
                x['BCUT2D_MRLOW']=pred_low
                pred.append(model.predict_proba(np.round(np.asarray(x))[0][1],2))     
            else:
                pred.append(f'Not_calculate')
        return pred
    
    #-------------------------------------------------------------------------------#
    def CARDIO_PREDICTION(smi,model):
        pred=[]
        for i in smi:
            x=utils.RDKIT_DESCRIPTORS([i])

            #Extract relavant feature
            x.drop(columns=['NumRadicalElectrons', 'Ipc', 'SMR_VSA8', 'SlogP_VSA9', 'fr_azide','fr_diazo', 'fr_isocyan', 'fr_isothiocyan', 'fr_nitro', 'fr_nitro_arom',
                           'fr_nitro_arom_nonortho', 'fr_nitroso', 'fr_prisulfonamd', 'fr_quatN','fr_thiocyan'],inplace=True)   #'AvgIpc'
            
            ##handle invalid smile
            if x.isnull().sum().sum()==194:
                pred.append(f'Not_calculate')
                continue
            ## Missing value handling
            X_data = cardio_KNN_IMPUTER.transform(x)
            # Scale the features using the pre-trained scaler
            scaled_features = cardio_MINMAX_SCALER.transform(X_data)

            if np.isnan(scaled_features).sum()==0:
                #booster = lgb.Booster(model_str=model)
                pred.append(np.round(model.predict_proba(scaled_features)[0][1],2))     
            else:
                pred.append(f'Not_calculate')
        return pred   
    
    #------------------------------------------------------------------------------------#
    def CARCINO_PREDICTION(smi,model):
        pred=[]
        for i in smi:
            x=utils.RDKIT_DESCRIPTORS([i])

            #Extract relavant feature
            #x.drop(columns=['Ipc'],inplace=True)   #'AvgIpc'
            
            ##handle invalid smile
            if x.isnull().sum().sum()>=50:
                pred.append(f'Not_calculate')
                continue
            ## Missing value handling
            X_data = x.fillna(0)

            ##load model
            model1,minmax_scaler=model
            # Scale the features using the pre-trained scaler
            scaled_features = minmax_scaler.transform(X_data)

            if np.isnan(scaled_features).sum()==0:
                #booster = lgb.Booster(model_str=model)
                pred.append(np.round(model1.predict(scaled_features)[0],2))     
            else:
                pred.append(f'Not_calculate')
        return pred        

    #----------------------------------------------------------------------------------#
    def MUTA_NEPHRO_NURO_PREDICTION(smi,model):
        pred=[]
        for i in smi:
            x=utils.ECFP4([i])
            if x.isnull().sum().sum()==0:
                pred.append(np.round(model.predict_proba(np.asarray(x))[0][1],2))
            else:
                pred.append('Not_calculate')
        return pred
    
    #---------------------------------------------------------------------------------#