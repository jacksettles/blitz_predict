from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from typing_extensions import Self
import matplotlib
from matplotlib.text import Text, Annotation
import numpy as np

@dataclass(frozen=True)
class FrameObject:  # 1 Kib +
    """Represents the x,y from ball/player in raw dataframe."""
    name: Optional[str]  # ball would be `None`
    x: int
    y: int
    on_offense: int
    blitz_probs: int
        
        
@dataclass
class PlayFrame:
    """Represents a frame of a play with a list of FrameObjects.
        Each FrameObject is a point/player on the field.
    """
    uuid: str = ""  # `{game_id}-{play_id}-{frame_id}`
    points: List[FrameObject] = field(default_factory=list)
        
        
@dataclass
class Play:
    """Represents a single play in a game with the frames being in order."""
    gpid: str = ""  # `{game_id}-{play_id}`
    frames: List[PlayFrame] = field(default_factory=list)
    off_team: str = ""
    def_team: str = ""
        
        
@dataclass
class _AnimArtists:
    off_scatter: matplotlib.collections.PathCollection | None = None
    def_scatter: matplotlib.collections.PathCollection | None = None
    ball_scatter: matplotlib.collections.PathCollection | None = None
    def_prob_texts: list[Text] = field(default_factory=list)
    def_name_annos: list[Text] = field(default_factory=list)
    def_face_rgba: np.ndarray | None = None
        