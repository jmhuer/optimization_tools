import re
import subprocess


def download_gdrive(share_link, local_name, print_stout=False):
  id = re.search('d/(.*?)/view', share_link).group(1)
  command = "wget --load-cookies /tmp/cookies.txt \"https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate \'https://docs.google.com/uc?export=download&id={}\' -O- | sed -rn \'s/.*confirm=([0-9A-Za-z_]+).*/\\1\\n/p')&id={}\"  -O \'{}\' && rm -rf /tmp/cookies.txt".format(id,id,local_name)
  returned_value = subprocess.run(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
  if print_stout: print(returned_value.stdout.decode("utf-8"))
  else: print("Download Complete: " + local_name)

