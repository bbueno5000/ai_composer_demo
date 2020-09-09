"""
DOCSTRING
"""
import nottingham_util
import rnn
import urllib
import zipfile

url = 'http://www-etud.iro.umontreal.ca/~boulanni/Nottingham.zip'
urllib.urlretrieve(url, 'dataset.zip')
zip = zipfile.ZipFile(r'dataset.zip') 
zip.extractall('data')
nottingham_util.create_model()
rnn.train_model()
