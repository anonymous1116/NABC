# 🔮 NABC: Neural Amortized Bayesian Computation (NABC)

This repository reproduces **"Neural Amortized Bayesian Computation (NABC)"**,  
a framework with three key components:

1. **Learning conditional mean function**  
2. **Learning conditional variance function**  
3. **Calibration**

For comparison, Neural Posterior Estimation (NPE) and Neural Likelihood Estimation (NLE) can be reproduced.
1. ** NPE and NLE training **
2. ** NPE and NLE inference **
   
---

## 📂 Repository Structure
### NABC
- `benchmark/benchmark_training.py` → trains the conditional mean function (**Sec 3.1**)  
- `benchmark/benchmark_cov_training.py` → trains the conditional variance function (**Sec 3.2**)  
- `benchmark/benchmark_calibrating.py` → performs calibration (**Sec 3.3**)  

### NPE and NLE
- `NPLE/NPE_training.py' → trains NPE and NLE method
- `NPLE/NPE_inference.py' → performs NPE and NLE inference 
---

## 🚀 NABC Usage

### 1️⃣ Train the conditional mean
```bash
python benchmark/benchmark_training.py \
    --num_training $num_training \
    --task $task \
    --N_EPOCHS $N_EPOCHS \
    --seed $seed \
    --layer_len $layer_len
```
- `$task ∈ {bernoulli_glm, my_twomoons, MoG_2, MoG_5, MoG_10, Lapl_5, Lapl_10, slcp_summary}`
- `$seed ∈ {1,…,10}`
- Defaults: `$layer_len=256`, `$N_EPOCHS=200`
- After this step, the folder **`NABC_nets/`** is created and the trained network is saved there.


### 2️⃣ Train the conditional variance
```bash
python benchmark/benchmark_cov_training.py \
    --num_training_mean $num_training_mean \
    --num_training_cov $num_training_cov \
    --task $task \
    --N_EPOCHS $N_EPOCHS \
    --seed $seed \
    --layer_len $layer_len
```
- `$num_training_mean` = value of `$num_training` from Step 1  
- `$num_training_cov` ≈ `2 × num_training_mean` (recommended)  
- `$seed ∈ {1,…,10}`  
- Defaults: `$layer_len=256`, `$N_EPOCHS=200`

### 3️⃣ Calibration
```bash
python benchmark/benchmark_calibrating.py \
    --x0_ind $x0_ind \
    --seed $seed \
    --L $L \
    --task $task \
    --num_training_mean $num_training_mean \
    --num_training_cov $num_training_cov \
    --layer_len $layer_len \
    --tol $tol
```
- $num_training_mean and $num_training_cov must match Step 2
- `$x0_ind` ∈ {1,…,10}, `$seed` ∈ {1,…,10}
- You can vary both `$L` and `$tol` where `$tol` corresponds to η in the manuscript



## 🚀 NPE and NLE Usage

### 1️⃣ NPE or NLE training
```bash
python NPLE/NPLE_training.py  --method $method \
 --task $task \
 --seed $seed \
 --cond_den $cond_den \
 --num_training $num_training 
```
- `$method` ∈ {NPE, NLE}
- `$seed` ∈ {1,…,10}
- `cond_den` ∈ {mdn, maf, nsf}

### 2️⃣ NPE or NLE inference
```bash
python NPLE/NPLE_training.py  --method $method \
 --cond_den $cond_den \
 --task $task \
 --seed $seed \
 --x0_ind $x0_ind \
 --num_training $num_training 
```
- `$x0_ind` ∈ {1,…,10}
