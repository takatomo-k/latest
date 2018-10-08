import subprocess
from collections import namedtuple

GPUState = namedtuple('GPUState', ['id', 'used_memory', 'total_memory'])

def get_device_meminfo() :
    result = subprocess.check_output(" nvidia-smi | awk '$9~/[0-9]+MiB/ {print $9, $11}'", shell=True)
    result = [[i]+list(map(float, [y.replace('MiB', '') for y in x.split()])) for i, x in enumerate(result.strip().split('\n'))]
    gpustates = []
    for rr in result :
        gpustates.append(GPUState(*rr))
    return gpustates

def get_most_free_device() :
    infos = get_device_meminfo()
    return sorted(infos, key=lambda x : x.used_memory)[0]
