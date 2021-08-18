Sentence Benchmark
==================

STS12, STS13, STS14, STS15, STS16, STS-B, SICK-R를 평가할 수 있는 벤치마크를 구성한다.

많은 부분 SentEval 레포지토리를 따라하지만 SentEval과 다르게 데이터셋을 쉽게 추가할 수 있게 만든다.

데이터는 무조건 \t으로 나뉘어져 있고 문장1\t문장2\t점수의 형태로 나타난다고 가정한다.

```bash
bash get_transfer_data.bash
conda create -n sentence-benchmark
conda activate sentence-benchmark
conda install --file requirements.txt
python -m evaluate
```


