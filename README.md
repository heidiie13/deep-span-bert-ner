# medical-deep-span-ner

- Create an environment:
```sh
conda create -n dsbert python=3.11.9 -y
conda activate dsbert
```

- Install dependencies (GPU):
```sh
conda install pytorch pytorch-cuda=12.1 -c pytorch -c nvidia
pip install -r requirements.txt
```

- Install dependencies (CPU):
```sh
conda install pytorch cpuonly -c pytorch
pip install -r requirements.txt
```