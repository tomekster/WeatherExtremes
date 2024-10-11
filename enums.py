from enum import Enum


class AGG(Enum):
    MEAN=1
    SUM=2
    MAX=3
    MIN=4
    
    
DAYS_PER_MONTH = [31,28,31,30,31,30,31,31,30,31,30,31]