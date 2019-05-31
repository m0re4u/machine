import torch
from itertools import product

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

OBJ_TYPES = ['box', 'ball', 'key']
COLORS = ['red', 'green', 'blue', 'purple', 'yellow', 'grey']
obj_list = list(product(COLORS, OBJ_TYPES))
mapping = {" ".join(k): v for v, k in enumerate(obj_list)}

def get_reason(obs, obs_info, index_only=False):
        """
        Extract a tensor containing the correctly reason label from the
        observation info.

        TODO: get label on type on instruction:
            - Before
            - After
            - And
            - Or

        if index_only is True, only give the index of the segment the agent
        should be aiming for. If False, give the index of the segment out of
        all possible color-object combinations.
        """
        task_status = [x['status'] for x in obs_info]
        res = [l.index('continue') for l in task_status]
        if index_only:
            return torch.as_tensor(res, device=device).type(torch.long)
        else:
            instructions = [x['mission'] for x in obs]
            split_instr = [i.split('then') for i in instructions]
            idx = [mapping[split[label].replace("go to the", "").strip()] for label, split in zip(res,split_instr)]
            return torch.as_tensor(idx, device=device).type(torch.long)