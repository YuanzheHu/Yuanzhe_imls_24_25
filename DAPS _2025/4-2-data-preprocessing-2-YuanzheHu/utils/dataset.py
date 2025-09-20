import pandas as pd
import numpy as np

import folktables
# Libraries for data preprocessing
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split


from abc import ABC, abstractmethod
 


class BaseDataset(ABC):
    
    @abstractmethod
    def acquire_dataset(self):
        pass


    def numerical_transforms(self,df):
        # Initialize MinMaxScaler
        mms = MinMaxScaler()

        # Fit and transform in one step
        df_num_mms = mms.fit_transform(df[self.numerical_features])

        # Create dataframe
        df_num_new = pd.DataFrame(df_num_mms, columns = self.numerical_features)

        # df_num_new.head()
        return df_num_new


    def categorical_transforms(self,df):
        # Initialize OneHotEncoder
        ohe = OneHotEncoder(handle_unknown='ignore', sparse = False)

        # Fit and transform in one step
        df_cat_ohe = ohe.fit_transform(df[self.categorical_features])

        # Create dataframe
        df_cat_new = pd.DataFrame(df_cat_ohe, columns = ohe.get_feature_names_out(self.categorical_features))

        return df_cat_new

    def data_transforms(self,df):
        # Transform numerical data
        df_num_new=self.numerical_transforms(df)

        # Transform categorical data
        df_cat_new= self.categorical_transforms(df)

        # Re-ensemble dataframe
        df_processed=pd.concat([df_num_new, df_cat_new, df[self.sensitive_group],df[self.target_task]], axis=1)
        
        return df_processed
    

    def generate_splits(self, df_trans, val_split=True):
        # Generate train and test sets
        train_data,test_data= train_test_split(df_trans, test_size=0.2, shuffle=True, random_state=3)
        
        if val_split:
            # Generate train and validation splits
            train_data,val_data= train_test_split(train_data, test_size=0.2, shuffle=True, random_state=3)
            # Print the shapes of the splits                      
            print('Validation data: ',val_data.shape)
            print('Train data: ',train_data.shape)
            print('Test data: ',test_data.shape)
            return train_data,val_data, test_data

        # Print the shapes of the splits
        print('Train data: ',train_data.shape)
        print('Test data: ',test_data.shape)
        return train_data, test_data


    def columnwise_partition(self, data):
        # Adjust format of training data
        y = data[self.target_task]
        sens= data[self.sensitive_group]
        data.drop(columns=[self.target_task,self.sensitive_group], inplace=True)
        X = data
        return X, sens, y

class ACSDataset(BaseDataset):
    def __init__(self, target_task,sensitive_group,state_list,duration,year,granularity):

        # Required features
        self.feature_columns = [
            'AGEP', # Age
            'SCHL', # Educational attainment
            'MAR', # Marital status
            'RELP', # Relationship
            'DIS', # Disability record
            'ESP', # Employment status of parents
            'CIT', # Citizenship status
            'MIG', # Mobility status
            'MIL', # Military service
            'ANC', # Ancestry code
            'NATIVITY', # Nativity - whether native or foreign born 
            'DEAR', # Hearing difficulty
            'DEYE', # Vision difficulty
            'DREM', # Cognitive difficulty
            'SEX', # Gender/Sex
            'RAC1P'] # Recorder detailed race code


        # Selected target - Employment status record
        self.target_task=target_task

        # Sensitive group
        self.sensitive_group=sensitive_group

        # Define numerical features
        self.numerical_features = ["AGEP"]# Age

        # Define categorical features
        self.categorical_features = [
            'SCHL', # Educational attainment
            'MAR', # Marital status
            'RELP', # Relationship
            'DIS', # Disability record
            'ESP', # Employment status of parents
            'CIT', # Citizenship status
            'MIG', # Mobility status
            'MIL', # Military service
            'ANC', # Ancestry code
            'NATIVITY', # nativity - whether native or foreign born 
            'DEAR', # Hearing difficulty
            'DEYE', # Vision difficulty
            'DREM', # Cognitive difficulty
            'SEX', # Gender/Sex
            'RAC1P'] # Recorder detailed race code

        # Define a race label dictionary    
        self.race_labels={
            1.0: 'White',
            2.0: 'Black or African American',
            3.0: 'American Indian',
            4.0: 'Alaska Native',
            5.0: 'American Indian and Alaska Native',
            6.0: 'Asian',
            7.0: 'Native Hawaiian and Other Pacific Islander',
            8.0: 'Some Other Race',
            9.0: 'Two or More Races'
            }

        # Create column name dictionary
        self.column_name_dict= {'AGEP': 'Age',
            'SCHL': 'Educational attainment',
            'MAR': 'Marital status',
            'RELP': 'Relationship',
            'DIS': 'Disability record',
            'ESP': 'Employment status of parents',
            'CIT': 'Citizenship status',
            'MIG': 'Mobility status',
            'MIL': 'Military service',
            'ANC': 'Ancestry code',
            'NATIVITY': 'Nativity - whether native or foreign born', 
            'DEAR': 'Hearing difficulty',
            'DEYE': 'Vision difficulty',
            'DREM': 'Cognitive difficulty',
            'SEX':'Sex',
                'RAC1P': 'Race'}
    
        # Pick a state to download data from available_states list.
        self.state_list=state_list

        # Pick data duration. Available options: ["1-Year", "5-Year"].
        self.duration=duration

        # Pick data year. Available options: ["2015", "2016", "2017", "2018"].
        self.year=year

        # Pick data granularity. Available options: ["person", "household"].
        self.granularity= granularity



    def acquire_dataset(self):
        
        # Define ML problem
        ACSEmployment = folktables.BasicProblem(
            features=self.feature_columns,
            target=self.target_task, 
            target_transform=lambda x: x == 1,    
            group=self.sensitive_group,
            preprocess=folktables.adult_filter,
            postprocess=lambda x: np.nan_to_num(x, -1),
        )

        # Construct the datasource 
        data_source = folktables.ACSDataSource(survey_year=self.year, horizon=self.duration, survey=self.granularity) 

        # Load data for the selected state
        acs_data = data_source.get_data(states=self.state_list, download=True) # Change to True to dowload, change to False not to

        # Define input features, target labels and sensitive groups and 
        features, label, group = ACSEmployment.df_to_numpy(acs_data)

        # Check correctness of splits
        assert len(features)== len(label)
        assert len(features)== len(group)

        # Change data structure to dataframe
        df=pd.DataFrame(
            np.concatenate((features, label.reshape(-1, 1)), axis=1),
            columns=self.feature_columns + [self.target_task],)
        
        return df
