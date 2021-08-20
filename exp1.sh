
method=$1

# Evaluate Noise Small

# Evaluate STS12
python -m evaluate \
    --method ${method} \
    --dataset STS12 \
    --data-dir data/STS/STS12-en-test

# Evaluate STS13
python -m evaluate \
    --method ${method} \
    --dataset STS13 \
    --data-dir data/STS/STS13-en-test

# Evaluate STS14
python -m evaluate \
    --method ${method} \
    --dataset STS14 \
    --data-dir data/STS/STS14-en-test

# Evaluate STS15
python -m evaluate \
    --method ${method} \
    --dataset STS15 \
    --data-dir data/STS/STS15-en-test

# Evaluate STS16
python -m evaluate \
    --method ${method} \
    --dataset STS16 \
    --data-dir data/STS/STS16-en-test
