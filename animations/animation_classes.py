from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from typing_extensions import Self

@dataclass(frozen=True)
class FrameObject:  # 1 Kib +
    """Represents the x,y from ball/player in raw dataframe."""
    id: Optional[str]  # ball would be `None`
    x: int
    y: int
    on_offense: int
    is_blitzing: int
        
        
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