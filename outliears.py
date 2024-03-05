import pandas as pd
import numpy as np



def run_eda(df: pd.DataFrame):
      """


      Perform EDA on the pandas dataframe
      Input: pd.DataFrame
      Output: display of main features of the dataframe


      """
      col_dtypes = pd.DataFrame(df.dtypes)
    
      numeric = []
      categorial = []
      strings = []
      other = []
      for n,type in enumerate(col_dtypes.iloc[:,0]):
              if (type == 'int64') |  (type == 'float64'):
                   if len(df.iloc[:, n].dropna().unique()) >= (df.shape[0]*0.75):
                       strings.append(df.columns[n])
                   else:
                       numeric.append(df.columns[n])
              if type == 'object':
                    if len(df.iloc[:, n].dropna().unique()) < df.shape[0]/2:
                       categorial.append(df.columns[n])

                    else:
                       strings.append(df.columns[n])
      for col in df.columns:
              if (col not in numeric) & (col not in categorial) & (col not in strings):
                  other.append(col)

      print('Analysis of numeric variables: ')
      print('----------------------------------')
      for n in numeric:
         print('For %s column' %n)
         print('mean=',np.nanmean(df[n]), ',  max=', np.max(df[n]), ',  min=', np.min(df[n]), ',  std=',np.nanstd(df[n]),
            ',  median=', np.nanmedian(df[n]),',  Q25=',np.nanpercentile(df[n], 25),  ',  Q75=',np.nanpercentile(df[n], 75), sep='')
         Q1 = df[n].quantile(0.25)
         Q3 = df[n].quantile(0.75)
         IQR = Q3 - Q1
         print('Number of outliers (± 1.5 IQR rule):', (len(df[n]) - sum(df[n].between(Q1, Q3, inclusive='both'))))
      #print('----------------------------------')
    #print('Analysis of categorial variables: ')
      #print('----------------------------------')
      for c in categorial:
              print('total number of unique values in %s column:' %c, len(df[c].dropna().unique()))
              print('frequencies of unique values','\n')
              freq_df = pd.DataFrame(df[c].value_counts() / len(df[c]))
              freq_df = freq_df.rename(columns={freq_df.columns[0]: 'frequency'})
              print(freq_df.to_string())



