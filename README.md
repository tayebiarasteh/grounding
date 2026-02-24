# TODO

This is the official repository of the paper [**TODO**](TODO).



## Environment setup

The implementation was based on Python 3.11.

### Prerequisites


```
$ conda create -n NAME python=3.11 -y
$ conda activate NAME
$ python -m pip install --upgrade pip
$ conda config --env --set channel_priority strict
$ conda install -y -c conda-forge pip
$ conda install -y -c conda-forge "numpy<2" "scipy>=1.10,<1.13" "scikit-learn>=1.3,<1.5" \
  pandas matplotlib tqdm jupyter pyarrow fastparquet
$ conda install -y -c conda-forge faiss-cpu
$ python -m pip install --upgrade pip
```


```
$ python -m pip install \
  torch==2.9.1 \
  torchvision==0.24.1 \
  torchaudio==2.9.1 \
  --index-url https://download.pytorch.org/whl/cu130
```

```
$ python -m pip install \
  accelerate \
  transformers \
  tokenizers \
  safetensors \
  huggingface-hub \
  timm \
  tensorboardX \
  opencv-python \
  opacus
```

```
$ conda install conda-forge::tiktoken
$ conda install conda-forge::sentencepiece
```

## Model initializations used


