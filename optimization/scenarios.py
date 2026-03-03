"""
Scenario Configurations for Supply Chain Optimization.

Defines meaningful, self-explanatory scenario IDs with parameter overrides
to support traceability, NLP integration, and sensitivity analysis.
"""

from typing import Dict, List
from dataclasses import dataclass


@dataclass
class ScenarioParameters:
    """Parameter multipliers for a specific scenario."""
    demand_multiplier: float = 1.0
    transport_cost_multiplier: float = 1.0
    lead_time_multiplier: float = 1.0
    safety_stock_multiplier: float = 1.0
    delay_probability_multiplier: float = 1.0


class ScenarioRegistry:
    """Registry of all available scenario configurations."""
    
    SCENARIOS: Dict[str, dict] = {
        "base_case_standard_conditions": {
            "description": "Normal forecast, normal transport cost, default risk penalties.",
            "category": "baseline",
            "parameters": ScenarioParameters(
                demand_multiplier=1.0,
                transport_cost_multiplier=1.0,
                lead_time_multiplier=1.0,
                safety_stock_multiplier=1.0,
                delay_probability_multiplier=1.0,
            ),
        },
        "risk_aware_high_disruption": {
            "description": "Increased delay probability and higher safety stock to simulate disruption risk.",
            "category": "risk",
            "parameters": ScenarioParameters(
                demand_multiplier=1.0,
                transport_cost_multiplier=1.0,
                lead_time_multiplier=1.2,
                safety_stock_multiplier=1.4,
                delay_probability_multiplier=1.8,
            ),
        },
        "cost_only_no_risk_penalty": {
            "description": "Optimization minimizes cost only, without risk penalties.",
            "category": "cost",
            "parameters": ScenarioParameters(
                demand_multiplier=1.0,
                transport_cost_multiplier=1.0,
                lead_time_multiplier=1.0,
                safety_stock_multiplier=0.5,
                delay_probability_multiplier=0.0,
            ),
        },
        "demand_spike_high_forecast": {
            "description": "Increased forecast values to simulate promotional or peak demand week.",
            "category": "demand",
            "parameters": ScenarioParameters(
                demand_multiplier=1.25,
                transport_cost_multiplier=1.0,
                lead_time_multiplier=1.0,
                safety_stock_multiplier=1.2,
                delay_probability_multiplier=1.0,
            ),
        },
        "transport_cost_increase_fuel_shock": {
            "description": "Increased shipping costs to simulate fuel price or logistics inflation.",
            "category": "supply_chain",
            "parameters": ScenarioParameters(
                demand_multiplier=1.0,
                transport_cost_multiplier=1.5,
                lead_time_multiplier=1.0,
                safety_stock_multiplier=1.0,
                delay_probability_multiplier=1.0,
            ),
        },
        "extended_lead_time_supplier_delay": {
            "description": "Increased lead times to simulate supplier or customs delays.",
            "category": "supply_chain",
            "parameters": ScenarioParameters(
                demand_multiplier=1.0,
                transport_cost_multiplier=1.0,
                lead_time_multiplier=1.5,
                safety_stock_multiplier=1.3,
                delay_probability_multiplier=1.2,
            ),
        },
    }
    
    @classmethod
    def get_scenario(cls, scenario_id: str) -> dict:
        """
        Retrieve scenario configuration by ID.
        
        Parameters
        ----------
        scenario_id : str
            The scenario identifier (e.g., 'base_case_standard_conditions')
        
        Returns
        -------
        dict
            Scenario configuration dict with 'description', 'category', and 'parameters'
        
        Raises
        ------
        ValueError
            If scenario_id is not found in registry
        """
        if scenario_id not in cls.SCENARIOS:
            available = ", ".join(cls.SCENARIOS.keys())
            raise ValueError(
                f"Unknown scenario '{scenario_id}'. Available scenarios: {available}"
            )
        return cls.SCENARIOS[scenario_id]
    
    @classmethod
    def get_scenario_safe(cls, scenario_id: str, default: str = "base_case_standard_conditions") -> dict:
        """
        Retrieve scenario configuration with fallback to default if not found.
        
        Parameters
        ----------
        scenario_id : str
            The scenario identifier
        default : str
            Default scenario ID to use if scenario_id not found
        
        Returns
        -------
        dict
            Scenario configuration dict
        """
        try:
            return cls.get_scenario(scenario_id)
        except ValueError:
            print(f"Warning: Unknown scenario '{scenario_id}'. Using default '{default}'.")
            return cls.get_scenario(default)
    
    @classmethod
    def list_scenarios(cls) -> Dict[str, str]:
        """
        List all available scenarios with descriptions.
        
        Returns
        -------
        dict
            Mapping of scenario_id → description
        """
        return {
            sid: config["description"] 
            for sid, config in cls.SCENARIOS.items()
        }
    
    @classmethod
    def list_scenarios_by_category(cls) -> Dict[str, List[str]]:
        """
        List scenarios grouped by category.
        
        Returns
        -------
        dict
            Mapping of category → list of scenario IDs
        """
        by_category = {}
        for sid, config in cls.SCENARIOS.items():
            category = config.get("category", "other")
            if category not in by_category:
                by_category[category] = []
            by_category[category].append(sid)
        return by_category
    
    @classmethod
    def get_parameters(cls, scenario_id: str) -> ScenarioParameters:
        """
        Get parameter multipliers for a scenario.
        
        Parameters
        ----------
        scenario_id : str
            The scenario identifier
        
        Returns
        -------
        ScenarioParameters
            Parameter multipliers for the scenario
        """
        config = cls.get_scenario_safe(scenario_id)
        return config.get("parameters", ScenarioParameters())
    
    @classmethod
    def load_scenario(cls, scenario_id: str) -> dict:
        """
        Load a scenario (convenience method for dashboard integration).
        Returns a dict with scenario metadata including description and parameters.
        
        Parameters
        ----------
        scenario_id : str
            The scenario identifier
        
        Returns
        -------
        dict
            Configuration dict with all scenario details
        """
        return cls.get_scenario_safe(scenario_id)


# Convenience aliases for backward compatibility
ScenarioConfig = ScenarioRegistry


__all__ = [
    "ScenarioRegistry",
    "ScenarioConfig",
    "ScenarioParameters",
]
