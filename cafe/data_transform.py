
#!/usr/bin/env python3

# Author: Justin Lin, M.S.

import warnings
warnings.filterwarnings('ignore')

import numpy as np 
import pandas as pd 

from Bio import SeqIO


def read_fasta(fasta_file):
    '''
    Reads in FASTA files

    Args:
        fasta_file: a FASTA file

    Returns a dataframe with the antibody's ID, sequence, and sequence length
    '''
    fasta_id = []
    fasta_sequences = []
    # Iterate through the FASTA file, extracting ID and sequence
    for seq_record in SeqIO.parse(fasta_file, 'fasta'):
        # Append the ID and the sequences to the empty lists
        fasta_id.append(seq_record.id)
        fasta_sequences.append(seq_record.seq)
    fasta_sequences = [''.join(x) for x in fasta_sequences]

    fasta_df = pd.DataFrame({'ID': fasta_id,
                            'Sequence': fasta_sequences})
    fasta_df['Sequence Length'] = fasta_df['Sequence'].astype(str).map(len)

    # Check if sequences are aligned
    assert all(seq_len == fasta_df['Sequence Length'][0] for seq_len in fasta_df['Sequence Length']), 'The sequences are not aligned. Please align the sequences using the software of your choice.'

    return fasta_df

def combine_seq(df1, df2):
    '''
    The function returns the entire/combined amino acid sequences given
    a dataframe containing the heavy chain sequences and another dataframe 
    containing the light chain sequences

    For example, if there are 60 heavy chain sequences and 20 light chain sequences, the combined 
    sequences results in 60 x 20 (1200) rows

    HC1: QVQLE  x   LC1: EVEL       =       QVQLEEVEL
                    LC2: EVEQ       =       QVQLEEVEQ
                    .                       .
                    .                       .
                    .                       .

    Args:
        df1: A dataframe
        df2: Another dataframe

    Returns a dataframe of the combined amino acid sequence
    '''
    # Initialize an empty list for the combined sequence
    comb_seq = []
    # For every heavy chain sequence, combine it with every light chain sequence
    for seq1 in df1:
        for seq2 in df2:
            combseq = seq1 + seq2
            # Append the combined sequence to the empty list
            comb_seq.append(combseq)

    # Create a dataframe of the combined sequence
    comb_df = pd.DataFrame({'Combined Sequence': comb_seq})

    return comb_df

def saar(df):
    '''
    Single Amino Acid Representation Transformation

    Once the sequences are combined, the function will split each amino acid into their own columns, 
    i.e., QVQLEEVEL -> Q V Q L E E V E L

    Args:
        df: A pandas dataframe containing the combined sequence

    Returns a dataframe of each amino acid from the combined sequence splits
    '''
    # Split each amino acid in the combined sequence into their own columns
    # and assign it to a new dataframe
    saar_df = df.apply(lambda x: pd.Series(list(x)))

    return saar_df

def convert_to_values(saar_df, values_file):
    '''
    Convert each amino acid to its respective numerical values based on
    a user provided Excel file

    Args:
        saar_df: The dataframe from saar()
        values_file: A Excel file

    Returns a dataframe of the amino acids converted to their respective
    numerical values
    '''
    values_df = pd.read_excel(values_file, index_col = 0)

    # Convert values_df to a dictionary to make it easier to replace the amino acids
    # with their respective values in hclc_split_df
    values_dict = values_df.to_dict()

    # Initialize an empty list that will contain all the different dataframes where
    # the amino acids have been replaced
    df_list = []
    for keys in values_dict.keys():
        transformed_df = saar_df.replace(values_dict[keys])
        df_list.append(transformed_df)

    sheet_names = [i for i in values_dict.keys()]

    return df_list, sheet_names

def parse_protparam(text_file):
    '''
    Parses the text file of protparam

    The function extracts the the input sequence name, the molecular weight value, the monoisotopic mass value,
    the theoretical pI value, the total number of negatively charged residues, the total number of positively
    charged residues, the instability index value, and the aliphatic index value

    Args:
        text_file: The text file of protparam (i.e. protparam.txt)

    Returns a dataframe of the extracted values
    '''
    # Read in the text file and remove new lines
    contents = [i.strip('\n') for i in open(text_file)]

    # Look for key phrases in contents
    key_phrases = ('Input sequence name:', 'Molecular weight: 2', 'Monoisotopic mass: 2', 'Theoretical pI:', 
                    'Total number of negatively', 'Total number of positively','The instability index (II)', 'Aliphatic index:')

    # If contents contain the key phrases, assign it to a list
    results = [b for b in contents if b.startswith(key_phrases)]

    # Iterate through results and create a nested list,
    # i.e., create a new list that starts with 'Input sequence name'
    # within results
    data = []
    for result in results:
        if 'Input sequence name' in result:
            data.append([result])
        else:
            data[-1].append(result)

    # Create a datframe from data list
    protparam_df = pd.DataFrame(data, columns = ['Input sequence name', 'Molecular weight', 'Monoisotopic mass', 'Theoretical pI', 
                                                'Total number of negatively charged residues', 'Total number of positively charged residues', 
                                                'Instability index', 'Aliphatic index'])

    # Extract the numbers from the text in each column except for 'Input sequence name'
    protparam_df['Input sequence name'] = protparam_df['Input sequence name'].str.replace('Input sequence name:', '')
    protparam_df['Molecular weight'] = protparam_df['Molecular weight'].str.extract(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?')
    protparam_df['Monoisotopic mass'] = protparam_df['Monoisotopic mass'].str.extract(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?')
    protparam_df['Theoretical pI'] = protparam_df['Theoretical pI'].str.extract(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?')
    protparam_df['Total number of negatively charged residues'] = protparam_df['Total number of negatively charged residues'].str.extract(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?')
    protparam_df['Total number of positively charged residues'] = protparam_df['Total number of positively charged residues'].str.extract(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?')
    protparam_df['Instability index'] = protparam_df['Instability index'].str.extract(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?')
    protparam_df['Aliphatic index'] = protparam_df['Aliphatic index'].str.extract(r'[+-]?(\d+(\.\d*)?|\.\d+)([eE][+-]?\d+)?')

    # Change the data types of all the columns
    protparam_df['Input sequence name'] = protparam_df['Input sequence name'].astype(str)
    protparam_df['Molecular weight'] = protparam_df['Molecular weight'].astype(float)
    protparam_df['Monoisotopic mass'] = protparam_df['Monoisotopic mass'].astype(float)
    protparam_df['Theoretical pI'] = protparam_df['Theoretical pI'].astype(float)
    protparam_df['Total number of negatively charged residues'] = protparam_df['Total number of negatively charged residues'].astype(float)
    protparam_df['Total number of positively charged residues'] = protparam_df['Total number of positively charged residues'].astype(float)
    protparam_df['Instability index'] = protparam_df['Instability index'].astype(float)
    protparam_df['Aliphatic index'] = protparam_df['Aliphatic index'].astype(float)

    return protparam_df
