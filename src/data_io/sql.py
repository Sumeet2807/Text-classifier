from data_io.base import Base_file
import pandas as pd
import pyodbc



class Pandas_inbuilt(Base_file):

    def read(self, args):

        cnxn_str = args['db-connection-string']
        cnxn = pyodbc.connect(cnxn_str)
        query = args['query']
        self.dataframe = pd.read_sql(query,cnxn)
        return self.dataframe

    def write(self,dataframe,args):

        
        cnxn_str = args['db-connection-string']
        table_name = args['table-name']
        cnxn = pyodbc.connect(cnxn_str)
        dataframe.to_sql(table_name,cnxn,if_exists='replace')
