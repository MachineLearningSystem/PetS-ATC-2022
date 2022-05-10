
import subprocess
import argparse
import string

# a~j
dst = string.ascii_lowercase[:10]

def upload():
    for d in dst:
        suffix = 'a'+d
        prefix = "mp_rank_00_model_states.pt"
        model_name = prefix+"."+suffix
        print("Uploading {}...".format(model_name))
        cmd = "osscmd --id=LTAIEWgqns5qyDNP --key=Eo8FGO3C83q5Y9VUGQ72LFTuOtj9QU \
        --host=oss-cn-hangzhou-zmf.aliyuncs.com \
        put {} oss://ait-public/hci_team/xuechao/model/".format(model_name)
        subprocess.Popen(cmd.split())
        
def download():
    for d in dst:
        suffix = 'a'+d
        prefix = "mp_rank_00_model_states.pt"
        model_name = prefix+"."+suffix
        print("Downloading {}...".format(model_name))
        cmd = "wget -c http://ait-public.oss-cn-hangzhou-zmf.aliyuncs.com/hci_team/xuechao/model/{}".format(model_name)
        subprocess.Popen(cmd.split())

def download2():
    for d in range(8):
        cmd = "wget -c https://alice-open.oss-cn-zhangjiakou.aliyuncs.com/sparse_plug/encoder/magnitude_0.99/mp_rank_0{}_model_states.pt".format(d)
        subprocess.Popen(cmd.split())

if __name__ == "__main__":
    download2()
