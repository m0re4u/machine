#!/usr/bin/env python3

"""
Evaluate a (partially) trained model or bot. Can also gather data
"""

import os
import time
import argparse

import gym
import torch
import numpy as np
import machine.util

from collections import defaultdict


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CONFUSION_DIR = 'confusions'

def reset_episode_data():
    episode_data = {
        'obs': [],
        'status': [],
        'embeddings': [],
        'prediction': []
    }
    return episode_data

def main(args):
    env = gym.make(args.env)

    # Set seed for all randomness sources
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    env.seed(args.seed)
    for _ in range(args.shift):
        env.reset()


    # Define agent
    partial = (args.reasoning == 'diagnostic' or args.reasoning == 'model')
    agent = machine.util.load_agent(
        env,
        args.model,
        env_name=args.env,
        vocab=args.vocab,
        partial=partial,
        diag_targets=args.diag_targets,
        drop_diag=args.drop_diag)

    # One process, two subtasks per process
    if "GoTo" in args.env:
        replace_instruction = r"go to the"
    elif "Pickup" in args.env:
        replace_instruction = r"pick up (the|a)"
    if "Transfer" in args.env:
        transfer_type = int(args.env.split("-")[1][-1])
    else:
        transfer_type = None
    reason_labeler = machine.util.ReasonLabeler(1,2, tt=transfer_type, replace_instr=replace_instruction)


    # Freeze layers and optionally load diagnostic model
    for name, param in agent.model.named_parameters():
        if 'reasoning' not in name:
            # Freeze layers we do not wish to train
            param.requires_grad = False
        if args.reasoning == 'diagnostic' and 'reasoning' in name and args.diag_model is not None:
            # Load trained diagnostic classifier
            state = torch.load(args.diag_model, map_location='cpu')
            param.data.copy_(state[name.partition('.')[2]])

    obs, info = env.reset()
    episode_terminations = defaultdict(lambda: 0)
    num_episodes = 0
    N_TARGETS = 18 if args.diag_targets is None else args.diag_targets
    num_frames = np.zeros((N_TARGETS, N_TARGETS))
    correct_frames = 0
    episode_data = reset_episode_data()
    episode_lengths = []
    ep_l = 0

    while True:
        time.sleep(args.pause)
        if args.show_gui:
            renderer = env.render("human")

        # Act with agent
        result = agent.act(obs)

        # Update the environment
        obs, reward, done, info = env.step(result['action'])
        ep_l += 1
        agent.memory *= (1 - done)

        # Save data for a frame in episode
        episode_data['obs'].append(obs)
        episode_data['status'].append(reason_labeler.annotate_status([obs], [info]))

        if args.gather:
            episode_data['embeddings'].append(agent.model.embedding.view(-1).cpu().numpy())
        _, pred_idx = result['reason'].max(1)
        episode_data['prediction'].append(pred_idx.item())

        if done:
            # Upon episode completion, get target reason
            target = reason_labeler.compute_reasons(
                torch.stack(episode_data['status']), episode_data['obs'])

            correct = torch.sum(torch.as_tensor(episode_data['prediction']).to(device).type(torch.int32) == target.to(device).flatten())
            correct_frames += correct.item()
            for i, pred in enumerate(episode_data['prediction']):
                num_frames[target[i].item()][pred] += 1
                if not args.machine:
                    print(f"Reason: {pred:2} - True: {target[i].item():2}")

            # Save data if we're gathering experience
            if args.gather:
                gather_data = np.array([episode_data['embeddings'], target.cpu().numpy().flatten()])
                np.save(f"data/{args.data_dir}/data_{num_episodes:03}", gather_data.T)

            # Check termination of episode
            if all([x == 'success' for x in info['status']]):
                # All objectives completed
                episode_terminations['success'] += 1
            elif any([x == 'success' for x in info['status']]) and 'or' in obs['mission'] and ep_l < env.max_steps:
                # one objective completed in "or" mission
                episode_terminations['success'] += 1
            elif reward == 0:
                # not completed, record why not
                if 'no_reward_reason' in info:
                    episode_terminations['task_failure'] += 1
                else:
                    episode_terminations['timeout'] += 1

            num_episodes += 1
            episode_lengths.append(ep_l)
            ep_l = 0
            if not args.machine:
                print(f"Completed mission {num_episodes:3}: {obs['mission']:50} - reward: {reward}")
            obs, info = env.reset()
            episode_data = reset_episode_data()

            if num_episodes > args.episodes:
                break
            else:
                continue

    for k, v in episode_terminations.items():
        episode_terminations[k]= v / num_episodes

    np.testing.assert_almost_equal(sum(episode_terminations.values()), 1, decimal=7, err_msg=f"-- {episode_terminations.values()}", verbose=True)

    # Print results
    if args.machine:
        #frames,acc,eplen,succ,fail,time
        print(f" {np.sum(num_frames)},{correct_frames / np.sum(num_frames)},{np.average(episode_lengths)},{episode_terminations['success']},{episode_terminations['task_failure']},{episode_terminations['timeout']}")
    else:
        print(f"\n\
                Reason accuracy:        {correct_frames / np.sum(num_frames)}\n\
                Average episode length: {np.average(episode_lengths)}\n\
                Success rate:           {episode_terminations['success']}\n\
                Failure rate:           {episode_terminations['task_failure']}\n\
                Timeout rate:           {episode_terminations['timeout']}\n\
                Frames observed:        {np.sum(num_frames)}")
    if args.confusion is not None:
        np.savetxt(f"{CONFUSION_DIR}/confusion_{args.confusion}.log", num_frames, fmt='%3.0f')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", required=True,
                        help="name of the environment to be run (REQUIRED)")
    parser.add_argument("--model", default=None, required=True,
                        help="name of the trained ACModel (REQUIRED)")
    parser.add_argument("--diag_model", default=None,
                        help="name of the trained diagnostic classifier")
    parser.add_argument("--diag_targets", default=None,type=int,
                        help="Number of outputs for diagnostic classifier")
    parser.add_argument("--drop_diag", default=False, action='store_true',
                        help="ignore loading of weights for diagnostic classifier")
    parser.add_argument("--vocab", default=None, required=True,
                        help="vocabulary file (REQUIRED)")
    parser.add_argument("--episodes", type=int, default=10,
                        help="number of episodes of evaluation (default: 10)")
    parser.add_argument("--seed", type=int, default=int(1e9),
                        help="random seed")
    parser.add_argument("--pause", type=float, default=0,
                        help="the pause between two consequent actions of an agent")
    parser.add_argument("--gather", default=False, action='store_true',
                        help="Whether to collect data for later training")
    parser.add_argument("--data_dir", default='reason_dataset',
                        help="directory name of collected data")
    parser.add_argument("--reasoning", type=str, default=None, choices=['diagnostic', 'model'],
                        help="Reasoning to ask the agent for")
    parser.add_argument("--machine", default=False, action='store_true',
                        help="print for machine use only")
    parser.add_argument("--show_gui", default=False, action='store_true',
                        help="Show a gui for human interpretation of agent actions")
    parser.add_argument("--shift", default=0, type=int,
                        help="Shift N episodes forward")
    parser.add_argument("--confusion", default=None, type=str,
                        help="Print the diagnostic classification confusion matrix for analysis to this file, None if disabled")

    args = parser.parse_args()

    # Make data saving directory if gathering experience hidden states
    if args.gather:
        DATA_DIR_NESTED = os.path.join('data',args.data_dir)
        if not os.path.isdir(DATA_DIR_NESTED):
            os.mkdir(DATA_DIR_NESTED)
    if args.confusion is not None:
        if not os.path.isdir(CONFUSION_DIR):
            os.mkdir(CONFUSION_DIR)

    main(args)
