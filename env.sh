conda create -n colmnew python=3.10
conda activate colmnew
conda install -c nvidia cuda-python
pip install -r requirement.txt --no-cache-dir --no-build-isolation
git clone https://github.com/hsgser/vllm.git
cd vllm
VLLM_INSTALL_PUNICA_KERNELS=1 pip install -e .
cd ..
pip install traker[fast] --no-cache-dir
pip install flash-attn==2.5.7 --no-build-isolation
pip install -i https://pypi.org/simple/ bitsandbytes
git clone https://github.com/decile-team/submodlib.git
cd submodlib
pip install -e .
cd ..
pip install -e .