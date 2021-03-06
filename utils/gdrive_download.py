import re
import subprocess


def download_gdrive(id, local_name, print_stout=True):
  coomand = 'gdown https://drive.google.com/uc?id={}'.format(id)
  returned_value = subprocess.run(coomand, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  if print_stout: print(returned_value.stdout.decode("utf-8"))
  else: print("Download Complete")

