from loading_utils.ngsim_dataset import I80Dataset, Us101Dataset
from loading_utils.highd_dataset import HighdLocations13Dataset, HighdLocations46Dataset


class DatasetTag:
    i80 = 0
    us101 = 1
    highd13 = 2
    highd46 = 3


DATASET_TAG2INFO = {
    DatasetTag.i80: I80Dataset,
    DatasetTag.us101: Us101Dataset,
    DatasetTag.highd13: HighdLocations13Dataset,
    DatasetTag.highd46: HighdLocations46Dataset,
}

