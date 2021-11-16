CLRCMD: a Contrastive Learning framework for Relaxed Contextualized token Mover's Distance
==================

```bash
cd src/scripts
bash get_transfer_data.bash
bash download_nli.sh
cd ../..
conda create -n clrcmd
conda activate clrcmd
conda install numpy scipy scikit-learn matplotlib
pip install -r requirements.txt
bash train_sts.sh
```
