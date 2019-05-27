#!/usr/bin/env python3

"""
Evaluate a (partially) trained model or bot. Can also gather data
"""

import argparse
import gym
import torch
import time
import numpy as np
import machine.util

def main(args):
    env = gym.make(args.env)

    # Set seed for all randomness sources
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)

    # Define agent and load in parts of the networks
    partial = (args.reasoning == 'diagnostic' or args.reasoning == 'model')
    agent = machine.util.load_agent(env, args.model, env_name=args.env, vocab=args.vocab, partial=partial)

    for name, param in agent.model.named_parameters():
        if 'reasoning' not in name:
            # Freeze layers we do not wish to train
            param.requires_grad = False
        if args.reasoning == 'reasoning' and 'reasoning' in name and args.diag_model is not None:
            # Load trained diagnostic classifier
            state = torch.load(args.diag_model)
            param.data.copy_(state[name.partition('.')[2]])

    loss_function = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(agent.model.parameters())

    obs, info = env.reset()
    num_episodes = 0
    num_frames = np.zeros(2)
    correct_frames = 0
    episode_data = []
    while True:
        time.sleep(args.pause)

        target = label_info(info['status'])
        result = agent.act(obs)

        # Update diagnostic classifier
        if args.train:
            loss = loss_function(result['reason'], target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        if args.gather:
            episode_data.append([agent.model.embedding.view(-1).numpy(),target.item()])

        # Check statistics
        num_frames[target] += 1
        _, pred_idx = result['reason'].max(1)
        print(f"Reason: {pred_idx.item()} - True: {target.item()}")
        if pred_idx.item() == target.item():
            correct_frames += 1

        # Perform action
        obs, reward, done, info = env.step(result['action'])
        agent.memory *= (1 - done)
        if done:
            # Upon episode completion
            if args.gather:
                np.save(f"data/reason_dataset/data_{num_episodes:03}", np.array(episode_data))
            num_episodes += 1
            print(f"Mission {num_episodes:3}: {obs['mission']:50} - reward: {reward}")
            obs, info = env.reset()
            if num_episodes > args.episodes:
                break
            else:
                continue

    # Print results
    print(f"\n\
            Accuracy:            {correct_frames / np.sum(num_frames)}\n\
            Frames observed:     {np.sum(num_frames)}\n\
            Frames for reason 0: {num_frames[0]}\n\
            Frames for reason 1: {num_frames[1]}")
    return

def label_info(info):
    """
    Get a string about the current progress of a level and format it to a
    one-hot vector for which reason we are training.
    """
    label = []
    for i, task in enumerate(info):
        if task == "continue":
            return torch.Tensor([i]).type(torch.long)



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True,
                        help="name of the environment to be run (REQUIRED)")
    parser.add_argument("--model", default=None, required=True,
                        help="name of the trained ACModel (REQUIRED)")
    parser.add_argument("--diag_model", default=None,
                        help="name of the trained diagnostic classifier")
    parser.add_argument("--vocab", default=None, required=True,
                    help="vocabulary file (REQUIRED)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="number of episodes of evaluation (default: 10)")
    parser.add_argument("--seed", type=int, default=int(1e9),
                        help="random seed")
    parser.add_argument("--pause", type=float, default=0,
                    help="the pause between two consequent actions of an agent")
    parser.add_argument("--train", default=False, action='store_true',
                    help="Whether to online train")
    parser.add_argument("--gather", default=False, action='store_true',
                    help="Whether to collect data for later training")
    parser.add_argument("--reasoning", type=str, default=None, choices=['diagnostic','model'],
                    help="Reasoning to ask the agent for")
    args = parser.parse_args()

    main(args)