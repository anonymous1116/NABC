# NABC: Neural Amortized Bayesian Computation (NABC)

This repository reproduces **"Neural Amortized Bayesian Computation (NABC)"**, a framework that combines three key components:

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
python benchmark/benchmark_training.py --num_training $num_training 
    --task $task 
    --N_EPOCHS $N_EPOCHS 
    --seed $seed 
    --layer_len $layer_len

For reproducibility, you can set 
$task is "bernoulli_glm", "my_twomoons", "MoG_2", "MoG_5", "MoG_10", "Lapl_5", "Lapl_10", "slcp_summary". You can vary $seed 1-10.
We fix $layer_len 256, $N_EPOCHS 200.

After implemeting this step, you will see the folder NABC_nets are generated, and the trained network is saved in the folder

### 2. Train the conditional variance
python benchmark/benchmark_cov_training.py 
    --num_training_mean $num_training_mean 
    --num_training_cov $num_training_cov 
    --task $task 
    --N_EPOCHS $N_EPOCHS 
    --seed $seed 
    --layer_len $layer_len

For $num_training_mean, you set the same number for $num_training above. $num_training_cov is usually set to be 2*$num_training_mean. You can vary $seed 1-10.
We fix $layer_len 256, $N_EPOCHS 200. 

### 3. Calibrate
python benchmark/benchmark_calibrating.py 
    --x0_ind $x0_ind 
    --seed $seed 
    --L $L 
    --task $task 
    --num_training_mean $num_training_mean 
    --num_training_cov $num_training_cov 
    --layer_len $layer_len 
    --tol $tol

Set $num_training_mean and $num_training_cov that are used in Step 2.

