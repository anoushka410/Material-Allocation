from pydantic import BaseModel
from typing import List, Optional

class CostImpact(BaseModel):
    transport_cost: Optional[float] = None
    holding_cost_change: Optional[float] = None
    stockout_penalty_avoided: Optional[float] = None
    net_cost_change: Optional[float] = None
    manufacturing_cost: Optional[float] = None
    distribution_cost: Optional[float] = None
    total_manufacturing_cost: Optional[float] = None

class ServiceLevelImpact(BaseModel):
    baseline_stockout_units: int
    post_transfer_stockout_units: int
    stockout_reduction_pct: float

class TransferRecommendation(BaseModel):
    scenario: str
    from_store: str
    to_store: str
    product_id: str
    quantity: int
    reason_codes: List[str]
    cost_impact: CostImpact
    service_level_impact: ServiceLevelImpact

class ManufacturingDecision(BaseModel):
    scenario: str
    product_id: str
    manufacture_quantity: int
    reason_codes: List[str]
    cost_impact: CostImpact

class ScenarioMetrics(BaseModel):
    total_cost: float
    total_stockouts: int
    total_transfers: Optional[int] = None
    manufacturing_units: Optional[int] = None

class ScenarioDelta(BaseModel):
    cost_change: float
    stockout_reduction_units: Optional[int] = None
    stockout_reduction_pct: float

class ScenarioSummary(BaseModel):
    scenario: str
    baseline: ScenarioMetrics
    optimized: ScenarioMetrics
    delta: ScenarioDelta
