import numpy as np
import pandas as pd

class Birthdays:
    def __init__(self, file):
        self.filepath=file
        self.table=None
        self.read_data()

    def read_data(self):
        f=open(self.filepath,'r')
        file=f.read()
        header,_,rows=file.partition('\n')
        numcolumns=header.count(',')+1
        columnnames=np.empty(numcolumns,dtype=np.dtypes.StrDType)
        remainingheader=header
        for i in range(numcolumns):
            current,_,remainingheader=remainingheader.partition(', ')
            columnnames[i]=current
        numrows=rows.count('\n')+1
        data=np.empty((numrows,numcolumns),dtype=np.dtypes.StrDType)
        self.table=pd.DataFrame(data,columns=columnnames)
        remainingrows=rows
        for i in range(numrows):
            currentrow,_,remainingrows=remainingrows.partition('\n')
            for j in range(numcolumns):
                currentelement,_,currentrow=currentrow.partition(', ')
                self.table.iloc[i, j]=currentelement