import subprocess

subprocess.run(['python3', '-m', 'textblob.download_corpora'])
subprocess.run(['python3', '-m', 'nltk.downloader', 'stopwords'])


print("Working")
