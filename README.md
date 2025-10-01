# NABC: Neural Amortized Bayesian Computation (NABC)

This repository reproduces **"Neural Amortized Bayesian Computation (NABC)"**,  
a framework that combines three key components:

1. **Learning conditional mean function**  
2. **Learning conditional variance function**  
3. **Calibration**

---

## 📂 Repository Structure
- `benchmark/benchmark_training.py` → trains the conditional mean function  (Sec 3.1)
- `benchmark/benchmark_cov_training.py` → trains the conditional variance function (Sec 3.2)
- `benchmark/benchmark_calibrating.py` → performs calibration (Sec 3.3)

---

## Usage

### 1. Train the conditional mean
python benchmark/benchmark_training.py \
    --num_training $num_training \
    --task $task \
    --N_EPOCHS $N_EPOCHS \
    --seed $seed \
    --layer_len $layer_len

### 2. Train the conditional variance
python benchmark/benchmark_cov_training.py \
    --num_training_mean $num_training_mean \
    --num_training_cov $num_training_cov \
    --task $task \
    --N_EPOCHS $N_EPOCHS \
    --seed $seed \
    --layer_len $layer_len

### 3. Calibrate
python benchmark/benchmark_calibrating.py \
    --x0_ind $x0_ind \
    --seed $seed \
    --L $L \
    --task $task \
    --num_training_mean $num_training_mean \
    --num_training_cov $num_training_cov \
    --layer_len $layer_len \
    --tol $tol
