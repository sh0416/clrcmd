Sentence Benchmark
==================

STS12, STS13, STS14, STS15, STS16, STS-B, SICK-R를 평가할 수 있는 벤치마크를 구성한다.

많은 부분 SentEval 레포지토리를 따라하지만 SentEval과 다르게 데이터셋을 쉽게 추가할 수 있게 만든다.

```bash
# SentEval benchmark data loading script
bash get_transfer_data.bash
# SimCSE training data loading script
wget -O .data/wiki1m_for_simcse.txt https://huggingface.co/datasets/princeton-nlp/datasets-for-simcse/resolve/main/wiki1m_for_simcse.txt
conda create -n sentence-benchmark
conda activate sentence-benchmark
conda install numpy scipy scikit-learn matplotlib
pip install -r requirements.txt
python -m run_evaluate
```

## Files

### Dataset
* `get_transfer_data.bash`: Download senteval data from web and preprocess it (e.g. mosestokenizer)
* `tokenizer.sed`: Tokenizer script used in `get_transfer_data.bash`

### Model
* `simcse/*`: Code related to the simcse paper
* `tokenizer.py`: Tokenizer that the bpe dropout is implemented in this script
* `run_train.py`: Finetune model using dataset

### Evaluate
* `run_evaluate.py` : Evaluate sentence representation quality using given benchmark data

