#!/bin/bash
# usage: scripts/research/ngc_image_env.sh <repo e.g. nvidia/vllm> <tag>  -> prints CUDA env of arm64 image config (nvcr.io anonymous)
repo=$1; tag=$2
tok=$(curl -s "https://nvcr.io/proxy_auth?scope=repository:$repo:pull" | python3 -c "import sys,json;print(json.load(sys.stdin)['token'])")
H=(-H "Authorization: Bearer $tok" -H "Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.docker.distribution.manifest.v2+json,application/vnd.oci.image.manifest.v1+json")
idx=$(curl -s "${H[@]}" "https://nvcr.io/v2/$repo/manifests/$tag")
dg=$(echo "$idx" | python3 -c "
import sys,json;d=json.load(sys.stdin)
for m in d.get('manifests',[]):
  if m['platform']['architecture']=='arm64': print(m['digest']);break
else: print('NOARM64' if 'manifests' in d else 'SINGLE')")
[ "$dg" = NOARM64 ] && { echo "$repo:$tag NO arm64"; exit; }
[ "$dg" = SINGLE ] && man="$idx" || man=$(curl -s "${H[@]}" "https://nvcr.io/v2/$repo/manifests/$dg")
cd=$(echo "$man" | python3 -c "import sys,json;print(json.load(sys.stdin)['config']['digest'])")
curl -sL "${H[@]}" "https://nvcr.io/v2/$repo/blobs/$cd" | python3 -c "
import sys,json;c=json.load(sys.stdin)
print('$repo:$tag', c.get('architecture'), [e for e in c['config'].get('Env',[]) if e.split('=')[0] in ('CUDA_VERSION','CUDA_DRIVER_VERSION','PYTHON_VERSION','NVIDIA_PYTORCH_VERSION','PYTORCH_VERSION','TRANSFORMERS_VERSION','VLLM_VERSION')])"
