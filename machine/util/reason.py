import torch
from itertools import product

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OBJ_TYPES = ['box', 'ball', 'key']
COLORS = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']
obj_list = list(product(COLORS, OBJ_TYPES))
mapping = {" ".join(k): v for v, k in enumerate(obj_list)}


class ReasonLabeler():
    def __init__(self, num_procs, num_subtasks):
        self.num_procs = num_procs
        self.num_subtasks = num_subtasks
        self.last_status = torch.zeros(self.num_procs, self.num_subtasks)
        self.prev_labels = torch.zeros(self.num_procs)

    def annotate_status(self, obs, obs_info):
        """
        Create tensor for every mission detailing the current status of the
        subtasks. This tensor will be used in the gradient updates when either
            1) determining the correct order of task completion
            2) determining the ratio of self-consistency

        Tensor has the form of an n-hot vector for every subtask in the mission,
        ones for every completed one.
        """
        task_status = [x['status'] for x in obs_info]
        hot_status = torch.zeros((len(obs_info), self.num_subtasks))
        for i, worker in enumerate(task_status):
            for j, subtask in enumerate(worker):
                hot_status[i][j] = int(subtask == 'success')
        return hot_status

    def compute_reasons(self, statuses, obs):
        """
        Backfill with correct reasons either due to 1) temporal ordering in
        subtasks or 2) self-consistency (correct label is first objective
        reached)
        """
        n_frames = statuses.size()[0]
        num_procs = statuses.size()[1]
        x = torch.zeros(n_frames, num_procs, dtype=torch.int)
        for i in reversed(range(n_frames)):
            prev_status = statuses[i-1, :, :] if i > 0 else self.last_status
            for p, proc in enumerate(prev_status):
                mission = obs[p]['mission']

                # Create label based on mission type
                if 'and' in mission:
                    x[i, p] = self.get_and_label(statuses[i, p, :], prev_status[p, :], mission)
                elif 'or' in mission:
                    pass
                elif 'then' in mission:
                    x[i, p] = self.get_then_label(statuses[i, p, :], mission)
                elif 'after' in mission:
                    x[i, p] = self.get_after_label(statuses[i, p, :], mission)

        # x should be n_frames by num_procs with either label of segment the
        # agent should predict or the object mapping
        self.last_status = statuses[-1, :, :].clone()
        return x

    def get_and_label(self, status, prev_status, mission):
        """

        """
        assert status.size() == prev_status.size()
        res = (prev_status != status).nonzero()
        label = 0
        if res.nelement() == 0:
            pass
        else:
            label = res[0].item()
        return label

    def get_then_label(self, status, mission):
        """
        Get reason label from a THEN (before) instruction. Since we have a strict
        temporal ordering we don't need any information besides the current
        subtask status and the instruction itself.
        """
        res = (status == 0).nonzero()[0].item()
        split_instr = mission.split('then')
        label = mapping[split_instr[res].replace("go to the", "").strip()]
        return label

    def get_after_label(self, status, mission):
        """
        Get reason label from an AFTER instruction. Since we again have a strict
        temporal ordering we don't need any information besides the current subtask
        status and the instruction itself.
        """
        res = (status == 0).nonzero()[-1].item()
        split_instr = mission.split('after')
        label = mapping[split_instr[res].replace("go to the", "").strip()]
        return label