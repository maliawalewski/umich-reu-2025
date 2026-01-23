Learning Monomial Orders in Grobner Basis Computations

This repo contains code and data for our Learning Monomial Orders in Groebner Basis Algorithms project, as well as code written during training for the project.

## Getting started

1) **Install Julia**

2) **Clone this repository**

3) **Run setup script to install packages:**

```bash
bash scripts/setup.sh
```

4) **To recreate tables/figures:**
```bash
bash scripts/make_figures.sh
```

5) **To train the RL agent:**
```bash
bash scripts/run_training.sh TRIANGULATION_BASE_SET 0 true false
```
or 
```bash
julia --project=. src/main.jl --baseset=TRIANGULATION_BASE_SET --seed=0 --PER=true --LSTM=false
```

### Flags for RL model:

| Flag | Type | Default |Description                                                                                
---|---|---|---
`‑‑baseset` | String | `N_SITE_PHOSPHORYLATION_BASE_SET` | Name of ideal baseset to use. It should be a variable from `basesets.jl` or `DEFAULT`.         
 `‑‑LSTM` | Bool   | `false`| `true` → use an LSTM for the actor; `false` → use a standard feed-forward neural network for the actor. 
 `‑‑PER` | Bool   | `true`| `true` → use prioritized experience replay; `false` → use uniform sampling replay buffer.
`--seed` | Int | `0` | RNG seed for reproducibility.

**Example:**

```bash
julia main.jl --baseset=DEFAULT --LSTM=false --PER=true --seed=0
```

### Flags for symbolic regression:
| Flag | Type | Default |Description                                                                                
---|---|---|---
`‑‑train` | Bool | `true` | `true` → train a symbolic regression model; `false` → skip training the symbolic regression model.         
`‑‑test` | Bool | `true` | `true` → evaluate the symbolic regression model; `false` → skip evaluating the symbolic regression model. 
`‑‑sr_mode` | String | `3target` | `3target` → learn an equation for each of the three weights; `2target` → learn equations for two weights and derive the third as `1 - w₁ - w₂`. 
`‑‑load_model` | Bool | `false` | `true` → load a symbolic regression model from `SR_MODEL_PATH` instead of training; `false` → do not load a model (train from scratch). 
`‑‑num_eval_ideals` | Int | `10` | Number of random ideal support sets used when testing the symbolic regression model. 
`‑‑eval_coefficients_per_ideal` | Int | `1000` | Number of random ideals (coefficients) per eval support set to test the model with. 


**Example:**
```bash
julia multi_target_sr.jl --train=true --test=true --sr_mode=3target
```

<img width="2048" height="2048" alt="Gemini_Generated_Image_te4y97te4y97te4y" src="https://github.com/user-attachments/assets/fb564f59-0fc2-445c-abf2-e31fe2289bf7" />
