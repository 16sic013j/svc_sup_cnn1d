import re


# nltk.download('stopwords')

def normalize_text(text):
    # Converting to Lowercase
    document = text.strip().lower()
    # remove all single characters
    document = re.sub(r'\s+[a-zA-Z]\s+', ' ', document)
    # Remove all digits
    document = re.sub(r'\b\d+\b', '', document)
    # Remove all the special characters
    document = re.sub(r'[ ](?=[ ])|[^-_,A-Za-z0-9 ]+', ' ', document)
    # Remove spaces
    document = re.sub(r'^\s+|\s+$|\s+(?=\s)', ' ', document)
    # Remove spaces
    document = re.sub(r'^\s+|\s+$|\s+(?=\s)', ' ', document)

    # remove stop words
    # stop_words = stopwords.words('english')

    return document


def preprocess(dataset_raw, preprocessfile_path):
    print('Preprocessing, please wait...')
    newfile = open(preprocessfile_path, 'w')
    for line in dataset_raw:
        temp = normalize_text(line)
        newfile.write(temp + '\n')
    newfile.close()
    print('Preprocessing done.')


def label2file(label, labelfile_path):
    print('creating label file, please wait...')
    newfile = open(labelfile_path, 'w')
    for line in label:
        line = str(line).strip()
        newfile.write(line + '\n')
    newfile.close()
    print('label file creation done.')
