os.chdir('C:\\Users\\Asus\\Downloads\\')
tsv_file = 'retention_indices_for_Afia.tsv'

# reading given tsv file
csv_table = pd.read_table(tsv_file, sep='\t')

# converting tsv file into csv
csv_table.to_csv('RI_HMDB_Afia.csv', index=False)

# output
print("Successfully made csv file")