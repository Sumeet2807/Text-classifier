import pandas as pd


class Base_file():

    def read(self,args):
        raise NotImplementedError

    def write(self,args):
        raise NotImplementedError