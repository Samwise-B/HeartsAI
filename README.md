# Reinforcement Learning in the Card Game Hearts

This project explores the application of **Reinforcement Learning (RL)** algorithms to multi-agent games with imperfect information, specifically through the card game Hearts. I implemented a **custom OpenAI Gym environment** for Hearts to facilitate training and benchmarking of the **Proximal Policy Optimization (PPO)** algorithm. This environment and study demonstrate my abilities in **deep reinforcement learning, environment modeling, and algorithmic optimization** within complex, multi-agent systems.

## Project Overview

### Goals
The objective of this research project was to evaluate the effectiveness of PPO in a custom-built environment simulating Hearts, a card game known for its multiple players, hidden information, sparse rewards, and complex state-action space. Key project goals included:
1. **Implementing the PPO Algorithm** for a custom Hearts environment to explore its viability.
2. **Benchmarking** the trained agents against rule-based agents and random agents.
3. **Optimizing Training Techniques** and reward mechanisms to improve agent performance.

### Research Summary
In this project, I evaluated how **policy gradient methods** like PPO perform in games with multi-agent, imperfect information. This project demonstrates a deep understanding of **RL challenges** such as non-stationary environments, sparse rewards, and partial observability. My approach involved encoding the game dynamics into a custom Gym environment that handles both the game's rules and the challenges of multi-agent interaction. The project ultimately produced several agent models that consistently outperformed random agents, highlighting the potential of PPO in complex games.

## Key Skills and Technologies Demonstrated

1. **Reinforcement Learning (RL) Implementation**: Applied and fine-tuned PPO, a popular policy gradient algorithm, for Hearts. PPO’s stability and efficiency in training are well-suited for multi-agent, imperfect-information scenarios like Hearts.
2. **Custom Environment Design (OpenAI Gym)**: Designed and implemented a custom Gym environment for Hearts, encoding game rules, state representation, and reward mechanisms to provide agents with a structured and consistent training ground.
3. **Neural Network Engineering**: Trained neural networks within the PPO framework to approximate value and policy functions effectively, achieving high sample efficiency and agent performance in this game context.
4. **Multi-Agent System Modeling**: Built an environment that accommodates multiple independent agents, capturing the unique dynamics of competitive card play in Hearts.
5. **Algorithm Optimization and Reward Engineering**: Experimented with different reward strategies to guide learning, addressing the challenge of sparse rewards by testing techniques that encourage positive agent behaviors in the game.
6. **Research and Evaluation**: Conducted detailed benchmarks and comparative analyses against rule-based and random agents, providing insights into the strengths and limitations of PPO in this domain.

## Installation and Setup

To replicate the environment and train agents, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/hearts-rl.git
   cd hearts-rl
2. See [SimpleAPI](https://github.com/davidADSP/SIMPLE) for futher details on training and testing the environment

# Key Findings and Results
- Improved Sample Efficiency: The custom reward structure improved sample efficiency, enabling agents to learn effective strategies faster.
- Successful Benchmarking: The PPO-trained agents consistently outperformed random agents, validating the feasibility of PPO in multi-agent, imperfect-information environments.
- Future Optimization: Potential avenues for future improvements include more advanced action/state representations and techniques to reduce exploration noise, which could further enhance agent performance.
# Future Work and Applications
This project establishes a foundation for applying reinforcement learning to complex, multi-agent games. Future applications include:

- Extending the environment to other multi-agent card games.
- Applying hierarchical reinforcement learning for more complex game scenarios.
- Exploring policy-sharing strategies among agents to further optimize decision-making.
- This project showcases a strong foundation in reinforcement learning, neural network training, and custom environment creation—skills applicable across various real-world AI challenges, particularly in multi-agent systems with incomplete information.

**Keywords**: Reinforcement Learning, Proximal Policy Optimization, OpenAI Gym, Multi-Agent Systems, Neural Networks, Artificial Intelligence
