import subprocess

import requests
import json

from pydub import AudioSegment
from tqdm import tqdm
import os

base_url = "https://www.audiolabs-erlangen.de"
base_page_url = base_url + "/resources/MIR/2017-GeorgianMusic-Scherbaum"
corpus_dir = "../data/GVM"

if os.path.exists(os.path.join(corpus_dir, "gvm_filepaths.json")):
    page_files = json.load(open(os.path.join(corpus_dir, "gvm_filepaths.json"), "r"))
else:
    base_page_html = requests.get(base_page_url).text
    pages = [row.split('"')[0] for row in base_page_html.split('id="myTable"')[1].split("</table>")[0].split('href="')][1:]
    page_files = {}
    for page in tqdm(pages):
        page_html = requests.get(base_url + page).text
        files = [file.split('"')[0] for file in page_html.split('<source src="')[1:]]
        page_files[page] = files
    json.dump(page_files, open(os.path.join(corpus_dir, "gvm_filepaths.json"), "w"))

file_list = ""
for name, url_suffixes in page_files.items():
    for url_suffix in url_suffixes:
        if ".mp3" in url_suffix:
            file_list += f"{base_url}{url_suffix}\n"
with open(os.path.join(corpus_dir, "filelist.txt"), "w") as f:
    f.write(file_list.strip())

os.chdir("../data/GVM")
subprocess.run("wget -i filelist.txt", shell=True, check=True)

files = [x for x in os.listdir() if ".mp3" in x]
for file in files:
    out_path = file.replace(".mp3", ".wav")
    AudioSegment.from_mp3(file).export(out_path, format="wav")
