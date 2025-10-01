# 🔮 NABC: Neural Amortized Bayesian Computation (NABC)

This repository reproduces **"Neural Amortized Bayesian Computation (NABC)"**,  
a framework with three key components:

1. **Learning conditional mean function**  
2. **Learning conditional variance function**  
3. **Calibration**

---

## 📂 Repository Structure
- `benchmark/benchmark_training.py` → trains the conditional mean function (**Sec 3.1**)  
- `benchmark/benchmark_cov_training.py` → trains the conditional variance function (**Sec 3.2**)  
- `benchmark/benchmark_calibrating.py` → performs calibration (**Sec 3.3**)  

---

## 🚀 Usage

### 1️⃣ Train the conditional mean
```bash
python benchmark/benchmark_training.py \
    --num_training $num_training \
    --task $task \
    --N_EPOCHS $N_EPOCHS \
    --seed $seed \
    --layer_len $layer_len
```
**Notes:**
- `$task ∈ {bernoulli_glm, my_twomoons, MoG_2, MoG_5, MoG_10, Lapl_5, Lapl_10, slcp_summary}`
- `$seed ∈ {1,…,10}`
- Defaults: `$layer_len=256`, `$N_EPOCHS=200`
- After this step, the folder **`NABC_nets/`** is created and the trained network is saved there.
