from enum import Enum


class SegmentType(Enum):
    COMMAND = 1
    OBJECT = 2
    PUTOBJECT = 3
    LOCSPEC = 4
    VERB = 5
    NOUN = 6
    PROP = 7
    ARTICLE = 8


words_to_wordtype = {
    'put': SegmentType.VERB,
    'go':  SegmentType.VERB,
    'pick': SegmentType.VERB,
    'up': SegmentType.PROP,
    'on': SegmentType.PROP,
    'in': SegmentType.PROP,
    'your': SegmentType.PROP,
    'open': SegmentType.VERB,
    'ball': SegmentType.NOUN,
    'key':  SegmentType.NOUN,
    'box':  SegmentType.NOUN,
    'door':  SegmentType.NOUN,
    'red':  SegmentType.NOUN,
    'green':  SegmentType.NOUN,
    'blue':   SegmentType.NOUN,
    'purple': SegmentType.NOUN,
    'yellow': SegmentType.NOUN,
    'grey':   SegmentType.NOUN,
    'next': SegmentType.PROP,
    'to':   SegmentType.PROP,
    'of':   SegmentType.PROP,
    'a': SegmentType.ARTICLE,
    'the': SegmentType.ARTICLE,
    'right': SegmentType.NOUN,
    'left': SegmentType.NOUN,
    'front': SegmentType.NOUN,
    'you': SegmentType.NOUN,
    'behind': SegmentType.NOUN,
    'then': SegmentType.PROP
}

colors = set([
    'red',
    'green',
    'blue',
    'purple',
    'yellow',
    'grey'
])


class SegmentHash():
    def __init__(self):
        self.max_size = 1000
        self.vocab = {}

    def __getitem__(self, key):
        if not (key in self.vocab.keys()):
            if len(self.vocab) >= self.max_size:
                raise ValueError("Maximum SegmentHash capacity reached")
            self.vocab[key] = len(self.vocab) + 1
        return self.vocab[key]


hasher = SegmentHash()


class Segment():
    def __init__(self, token, type):
        self.words = token.split()
        self.type = type

    def __repr__(self):
        sent = "_".join(self.words)
        return sent

    def __hash__(self):
        return hash(str(self.words))

    def __eq__(self, other):
        _words = self.words == other.words
        _type = self.type == other.type
        return _words and _type

    def __format__(self, fmt):
        return format(str(self), fmt)


class Segmenter():
    def __init__(self, mission, segment_level='word'):
        self.LEVELS = [
            'word',
            'word_annotated',
            'segment'
        ]
        self.segment_level = segment_level
        self.mission = mission
        if self.segment_level not in self.LEVELS:
            raise NotImplementedError(
                f"Incorrect segment level, pick out of {self.LEVELS}")

    def segment(self, instruction):
        instruction_list = instruction.split()
        if self.segment_level == 'word':
            return [str(Segment(x, words_to_wordtype[x])) for x in instruction_list]
        elif self.segment_level == 'segment':
            # Rule based segmentation based on level
            segs = self.rule_segment(instruction_list)
            return [str(seg) for seg in segs]
        elif self.segment_level == 'word_annotated':
            segs = self.rule_segment(instruction_list, annotate_only=True)
            return [str(word) + "_" + str(hasher[seg]) for seg in segs for word in seg.words]

    def rule_segment(self, instruction_list, annotate_only=False):
        if self.mission == "BabyAI-PutNextLocal-v0":
            cmd = instruction_list[:1]
            obj1 = instruction_list[1:4]
            locobj = instruction_list[4:6]
            obj2 = instruction_list[6:]
            return [
                Segment(" ".join(cmd), SegmentType.COMMAND),
                Segment(" ".join(obj1), SegmentType.OBJECT),
                Segment(" ".join(locobj), SegmentType.PUTOBJECT),
                Segment(" ".join(obj2), SegmentType.OBJECT)
            ]
        elif (self.mission == "BabyAI-GoToLocal-v0" or
                self.mission == "BabyAI-GoTo-v0"):
            cmd = instruction_list[:2]
            obj1 = instruction_list[2:]
            return [
                Segment(" ".join(cmd), SegmentType.COMMAND),
                Segment(" ".join(obj1), SegmentType.OBJECT)
            ]
        elif self.mission == "BabyAI-PickupLoc-v0":
            if len(colors & set(instruction_list)):
                # colored object
                obj_e = 5
            else:
                # no color specifier for object
                obj_e = 4
            cmd = instruction_list[:2]
            obj1 = instruction_list[2:obj_e]
            loc = instruction_list[obj_e:]
            if loc != []:
                return [
                    Segment(" ".join(cmd), SegmentType.COMMAND),
                    Segment(" ".join(obj1), SegmentType.OBJECT),
                    Segment(" ".join(loc), SegmentType.LOCSPEC)
                ]
            else:
                return [
                    Segment(" ".join(cmd), SegmentType.COMMAND),
                    Segment(" ".join(obj1), SegmentType.OBJECT)
                ]
