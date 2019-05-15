#!/usr/bin/env python3

"""
Evaluate a trained model or bot
"""

import argparse
import gym
import torch
import time
import numpy as np
import machine.util
from babyai.evaluate import evaluate_demo_agent, batch_evaluate, evaluate

def main(args):

    # Set seed for all randomness sources
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)

    # Define agent
    env = gym.make(args.env)
    env.seed(args.seed)

    agent = machine.util.load_agent(env, args.model, env_name=args.env, vocab=args.vocab)

    obs = env.reset()
    while True:
        time.sleep(args.pause)
        result = agent.act(obs)
        obs, reward, done, _ = env.step(result['action'])
        if done:
            print(f"Mission: {obs['mission']:50} - reward: {reward}")
            obs = env.reset()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True,
                        help="name of the environment to be run (REQUIRED)")
    parser.add_argument("--model", default=None,
                        help="name of the trained model (REQUIRED)")
    parser.add_argument("--vocab", default=None, required=True,
                    help="vocabulary file (REQUIRED)")
    parser.add_argument("--episodes", type=int, default=1000,
                        help="number of episodes of evaluation (default: 1000)")
    parser.add_argument("--seed", type=int, default=int(1e9),
                        help="random seed")
    parser.add_argument("--pause", type=float, default=0.1,
                    help="the pause between two consequent actions of an agent")
    args = parser.parse_args()

    main(args)