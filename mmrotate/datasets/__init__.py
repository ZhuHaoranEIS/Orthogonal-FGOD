# Copyright (c) OpenMMLab. All rights reserved.
from .builder import build_dataset  # noqa: F401, F403
from .dota import DOTADataset  # noqa: F401, F403
from .hrsc import HRSCDataset  # noqa: F401, F403
from .pipelines import *  # noqa: F401, F403
from .sar import SARDataset  # noqa: F401, F403

from .fair1m import FAIR1MDataset, FAIR1MCourseDataset
from .shiprs import ShipRSImageNet
from .mar20 import MAR20Dataset
from .fair1m_confusion import CONFUSIONFAIR1MDataset
from .dior_scene import DIOR_SCENEDataset

__all__ = ['SARDataset', 'DOTADataset', 'build_dataset', 'HRSCDataset', 'FAIR1MDataset',
           'FAIR1MCourseDataset', 'ShipRSImageNet', 'MAR20Dataset', 'build_dataset', 'CONFUSIONFAIR1MDataset',
           'DIOR_SCENEDataset', ]
