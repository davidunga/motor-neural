
from typechecking import *
from neural import NeuralData
from motor import KinData
from dataclasses import dataclass

# ----------------------------------


@dataclass
class DatasetMeta:
    """ Dataset metadata """
    name: str
    task: str
    monkey: str
    sites: Set[str]
    file: str


# ----------------------------------


@dataclass
class Trial:
    """ Single trial data """

    dataset: str
    ix: int
    lag: float
    bin_sz: float

    kin: KinData = None
    neural: NeuralData = None

    _events: dict[str, float] = None
    _properties: dict[str, Any] = None

    def __post_init__(self):
        if self._events is not None:
            self.set_events(self._events)
        if self._properties is not None:
            self.set_properties(self._properties)
        pass

    @property
    def duration(self):
        return self.end - self.st

    def get_events(self):
        return self._events

    def set_events(self, events_dict: dict[str, float]) -> None:
        """ Validate format and order by event times """
        if "st" not in events_dict or "end" not in events_dict:
            raise ValueError("Events must include 'st' (trial start time) and 'end' (trial end time)")
        assert is_type(events_dict, dict[str, float])
        keys = list(events_dict.keys())
        values = list(events_dict.values())
        if not set(keys).isdisjoint(set(self._properties.keys())):
            raise ValueError("Event names clash with existing properties")
        self._events = dict({keys[ix]: values[ix] for ix in np.argsort(values)})

    def add_events(self, events_dict: dict[str, float]) -> None:
        for k, v in self._events.items():
            if k in events_dict:
                raise ValueError(f"Event {k} already exists")
            events_dict[k] = v
        self.set_events(events_dict)

    def set_properties(self, properties_dict: dict[str, Any]) -> None:
        if not set(properties_dict.keys()).isdisjoint(set(self._events.keys())):
            raise ValueError("Property name clash with existing event names")
        self._properties = properties_dict

    def __getattr__(self, item):
        return self.__getitem__(item)

    def __getitem__(self, item):
        if item in self._events:
            return self._events[item]
        elif item in self._properties:
            return self._properties[item]
        else:
            raise AttributeError(f"Unknown attribute " + item)
        pass


# ----------------------------------

@dataclass
class Data:

    """ Highest level data container
        Behaves as a Trial iterator, and provides access to dataset-level attributes
    """

    def __init__(self, trials: list[Trial], meta: DatasetMeta):
        self._trials = trials
        self._meta = meta
        self._validate()

    def _validate(self):
        if not len(self._trials):
            raise ValueError("Cannot initialize data with empty trial list")
        # properties that are suppose to be the same for all trials:
        assert all([tr.lag == self._trials[0].lag for tr in self._trials])
        assert all([tr.bin_sz == self._trials[0].bin_sz for tr in self._trials])

    def set_trials(self, trials: list[Trial]):
        self._trials = trials
        self._validate()

    @property
    def meta(self) -> DatasetMeta:
        return self._meta

    @property
    def lag(self) -> float:
        return self._trials[0].lag

    @property
    def bin_sz(self) -> float:
        return self._trials[0].bin_sz

    @property
    def num_neurons(self) -> int:
        return self._trials[0].neural.num_neurons

    def __str__(self):
        return f"{self.name}: {len(self)} trials, {self.num_neurons:d} neurons, " \
               f"lag={self.lag:2.2f}s, bin={self.bin_sz:2.2f}s"

    def __iter__(self):
        yield from self._trials

    def __next__(self):
        return self.__iter__().__next__()

    def __len__(self):
        return len(self._trials)

    def __getitem__(self, item):
        if isinstance(item, np.ndarray):
            if item.dtype == bool:
                item = np.nonzero(item)[0]
            assert item.dtype == int
            return [self._trials[ix] for ix in item]
        return self._trials[item]

    def __getattr__(self, item):
        return self._meta.__getattribute__(item)

