# Deep Q-Learning for Pong

#### **Overview**

This project implements a Deep Q-Network (DQN) to teach an artificial agent how to play Atari Pong from raw pixels using reinforcement learning.

**The code follows:**

- Clean modular structure
- Industry-standard coding practices
- Full logging (CSV), checkpointing, and safety checks
- Multiple experiment configurations
- Automatic comparison plots

The agent sees the game as stacked grayscale frames and learns using experience replay and target networks — the classic DQN algorithm.

#### Project Folder Structure


RL_Assignment3/
│
├── model.py                  # CNN DQN architecture
├── agent.py                  # Epsilon-greedy action selection agent
├── replay_buffer.py          # Experience replay memory
├── train.py                  # Training loop, evaluation, logging, checkpointing
├── run_experiments.py        # Runs 4 experiment configs + comparison plot
├── assignment3_utils.py      # Frame preprocessing + reward shaping
│
├── results/                  # Auto-generated logs, plots, and checkpoints
│   ├── batch8_target10_rewards.npy
│   ├── batch16_target10_rewards.npy
│   ├── batch8_target3_rewards.npy
│   ├── batch16_target3_rewards.npy
│   ├── comparison_plot.png
│   ├── *_reward_plot.png
│   ├── *_training_log.csv
│   └── dqn_checkpoint.pth
│
├── README.md                 # ← (this file)
└── requirements.txt          # Python dependencies

#### 🚀 Features

✔ Modular design (clear separation of model, agent, buffer, training)
✔ Checkpoint saving + auto resume
✔ CSV logging for losses, rewards, epsilon
✔ Safe environment resets & step handling
✔ Replay buffer size checks
✔ Four experiment configurations:
  - Batch 8 / Target 10
  - Batch 16 / Target 10
  - Batch 8 / Target 3
  - Batch 16 / Target 3
✔ Automatic combined performance plot

#### Algorithm Summary 

- The agent sees 4 game frames at a time.
- A neural network predicts Q-values (goodness of moves).
- The agent sometimes explores (random moves) and sometimes exploits (best move).
- Experiences are stored in the replay buffer to train from random past events.
- A “target network” stabilizes learning by updating slowly.

Over 500 episodes, the agent gets slightly better at Pong.

#### 🛠 Installation

**1. Clone this repository:**

git clone this repo
cd RL_Assignment3

**2. Create and activate a virtual environment:**

python -m venv venv
source venv/bin/activate      # macOS/Linux
venv\Scripts\activate         # Windows

**3. Install dependencies:**

pip install -r requirements.txt

4. Install Gym Atari support:
pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]

##### ▶️ How to Run the Experiments

Run all 4 experiments automatically:

`python run_experiments.py`


**This will:**

- Train 4 models
- Save plots, logs, and checkpoints
- Produce a comparison plot in results/comparison_plot.png

##### 📊 Output Files

**Inside results/, you will find:**

File	

#### Experiment Configurations

Each experiment varies batch size and target update frequency:

Experiment	Batch Size	Target Update	Purpose
A	8	10	Baseline: stable, lighter updates
B	16	10	Larger batch → smoother learning
C	8	3	Faster target updates (more unstable)
D	16	3	Large batch + fast updates

Running all 4 helps understand how these hyperparameters affect stability.

#### 📈 Expected Results (Based on Assignment Requirements)

Your agent will:

Start scoring -21 to -20 (random play)

Slowly reach -17 to -16 average

Not become good at Pong within 500 episodes
(normal DQN takes > 1,000 to 10,000 episodes)

Example Partial Log:
Ep 000 | Reward: -20.0 | Avg5: -20.00 | Eps: 1.00
Ep 010 | Reward: -21.0 | Avg5: -20.60 | Eps: 0.05
Ep 200 | Reward: -19.0 | Avg5: -18.00 | Eps: 0.05
Ep 210 | Reward: -16.0 | Avg5: -17.40 | Eps: 0.05

####  Code Explanation (High-Level)

**1.model.py**

Defines DQNCNN — a CNN matching the original DeepMind architecture.

**2.agent.py**

Handles epsilon-greedy action selection.

**3.replay_buffer.py**

Stores experiences and samples random batches.

**4.train.py**

Core training loop:

gameplay

replay buffer updates

gradient updates

target network sync

checkpoint saving

CSV logging

run_experiments.py

Runs all experiment configurations, saves plots, and generates comparison charts.

🐞 Troubleshooting
Pong not found?

Install Atari ROMs:

pip install gymnasium[atari]
pip install gymnasium[accept-rom-license]

Black screens or shape errors?

Likely preprocessing mismatch → confirm frame shape = (84, 80).

Loss becomes NaN?

The code already includes:

if torch.isnan(loss): skip update


This prevents crashes.