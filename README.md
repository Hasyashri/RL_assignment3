
# Folder Structure 

Assignment3/
│
├── assignment3_utils.py        # (given to you)
├── replay_buffer.py            # implements experience replay
├── model.py                    # defines the DQN CNN
├── agent.py                    # defines DQNAgent class
├── train_pong_dqn.py           # main training script
├── evaluate_and_plot.py        # script for generating plots and comparisons
└── report_template.md          # report text (convert to PDF)


# Step-by-Step Implementation Plan

1️⃣ **replay_buffer.py**

Implements the ReplayBuffer class.
Handles experience storage and random sampling.

2️⃣ **model.py**

Implements the CNN-based DQN model using PyTorch:

3 Conv layers, 2 Fully connected layers

Dynamic computation of convolution output

Outputs Q-values for 6 actions

3️⃣ **agent.py**

Implements:

DQNAgent class

ε-greedy action selection

Learning step with Bellman target

Target network update every N episodes

Epsilon decay

4️⃣ **train_pong_dqn.py**

Handles:

Environment creation (PongDeterministic-v4)

Preprocessing via assignment3_utils.py

Frame stacking (4 frames as one state)

Training loop over episodes

Logging (episode, score, avg reward, epsilon)

Model checkpointing

5️⃣ **evaluate_and_plot.py**

Generates:

Episode vs Score

Episode vs Average Reward (last 5)
Plots for:

Batch size variation: [8, 16]

Target update frequency: [3, 10]

6️⃣ **report_template.md**

Markdown report containing:

Introduction (about DQN, Pong, motivation)

Methodology (preprocessing, CNN architecture)

Experiments (batch size and target update ablations)

Results & Discussion (plots and insights)

Conclusion (best configuration and justification)