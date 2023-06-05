import itertools as it
import logging
import json
from copy import deepcopy
from typing import List, Literal, Union, ClassVar, Dict, Optional
from pathlib import Path
from functools import cache
from enum import Enum

from pydantic import BaseModel, validator, PositiveInt, ValidationError

logger = logging.getLogger(__name__)

DATASET_META_LOCATION = Path(__file__).parent.parent / "datasets"
DATASETS = [
    file.stem
    for file in filter(
        lambda path: path.is_file() and path.suffix == ".json",
        DATASET_META_LOCATION.iterdir(),
    )
]


class Files(BaseModel):
    input_dir_source: Path
    input_dir_target: Path
    invars_source: Optional[Path]
    invars_target: Optional[Path]
    output_dir: Path

    @validator("input_dir_source", "input_dir_target", "output_dir")
    def valid_directory(cls, v):
        if not v.is_dir():
            raise ValueError(f"Directory {v} does not exist.")

        return v

    @validator("invars_source", "invars_target")
    def valid_file(cls, v):
        if not (v is None or v.is_file()):
            raise ValueError(f"Invariants file {v} does not exist.")

        return v


class TimeDomain(BaseModel):
    season_map: ClassVar[Dict[str, List[int]]] = {
        "DJF": [12, 1, 2],
        "MMA": [3, 4, 5],
        "JJA": [6, 7, 8],
        "SON": [9, 10, 11],
        "summer": [4, 5, 6, 7, 8, 9],
        "winter": [10, 11, 12, 1, 2, 3],
        "all": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    }

    years: List[int]
    months: List[int]

    @validator("month", pre=True)
    def parse_season(cls, v):
        try:
            return cls.season_map[v]
        except KeyError:
            logger.debug(f"{v}: is not a valid season")
            return v

    @cache  # TODO confirm cache works on property
    @property
    def year_months(self):
        return list(it.product(self.years, self.months))

    # TODO implement subset/superset interface


class Interpolation(BaseModel):
    start: PositiveInt
    end: PositiveInt


class Interpolations(BaseModel):
    p: Union[None, Interpolation]
    z: Union[None, Interpolation]


class Variable(BaseModel):
    prefix_map: ClassVar[Dict[str, str]] = {
        "surface": "sf",
        "const": "c",
        "m_lvl": "m",
        "p_lvl": "p",
        "z_lvl": "z",
    }
    name: str
    # TODO: use proper lvl type
    # ??? const => surface ???
    type: Literal["surface", "const", "m_lvl", "p_lvl", "z_lvl"]
    lvls: List[int]

    @property
    def ml_names(self):
        # !! unit (hpa, pa, m, km) ? {"t", "p_lvl", 85000} => t850
        try:
            names = [f"{self.name}{lvl}" for lvl in self.lvls]
        except TypeError:
            names = [self.name]
        return names

    @validator("lvls")
    def check_lvls(cls, values, v):
        ml_variables = ["m_lvl", "p_lvl", "z_lvl"]

        if values["type"] in ml_variables:
            be_empty = False
        else:
            be_empty = True

        is_empty = len(v) == 0

        if be_empty ^ is_empty:  # xor
            raise ValueError(
                f"multilvl vars expect at least 1 lvl, 2d vars should have no lvl"
            )

        return v

    @validator("lvls", pre=True)
    def convert_none(cls, v):
        if v is None:
            return []
        return v


class GridAxis(BaseModel):
    name: str
    longname: str
    size: int
    units: Literal["degrees"]
    first: Optional[float]  # for limited area domains
    inc: float


class CDOGrid(BaseModel):
    gridtype: Literal["lonlat", "gaussian", "projection", "curvilinear", "unstructured"]
    gridsize: int
    x: GridAxis
    y: GridAxis


# Domain without validation for domain of dataset
class InfoDomain(BaseModel):
    dataset: str
    variables: List[Variable]
    time: TimeDomain
    grid: CDOGrid  # TODO: make sure also CDOProjectionGrid gets parsed

    def get_type(self, type: str):
        return list(filter(lambda var: var.type == type, self.variables))

    def lvls(self, type: str):
        return {*it.chain(*(var.lvl for var in self.get_type(type)))}

    def subset(self, variables=None, time=None, grid=None):
        return Domain(
            self.dataset,
            list(
                filter(lambda var: var in self.variables, variables)
            ),  # TODO make variables comparable
            self.time.subset(time),
            self.grid.subset(grid),
        )

    @property
    def p_lvls(self):
        # TODO: use proper lvl type (enum)
        return self.get_lvls("p_lvl")

    @property
    def z_lvls(self):
        # TODO: use proper lvl type (enum)
        return self.lvls("z_lvls")

    @property
    def var_names(self):
        return [var.name for var in self.variables]

    @property
    def ml_names(self):
        return [var.ml_names() for var in self.variables]


class Domain(InfoDomain):  # TODO: check if domain available in 'dataset'
    @validator("dataset")
    def dataset_avail(cls, v):
        try:
            get_info(v)
        except ValueError as e:
            raise e
        return v

# describes one dataset
class DatasetInfo(BaseModel):
    """
    Describes available domain/possible interpolations.
    """

    has_constants: bool
    domain: InfoDomain
    interpolation: Interpolations


@cache
def get_info(name: str) -> DatasetInfo:
    file = DATASET_META_LOCATION / f"{name}.json"
    try:
        with open(file, "r") as f:
            return DatasetInfo(**json.load(f))
    except FileNotFoundError as e:
        raise ValueError(f"Cannot access {name} information: {f} not available")
    except ValidationError as e:
        raise ValueError(
            f"Cannot access {name} information: Invalid Format of {f}\n{str(e)}"
        )
