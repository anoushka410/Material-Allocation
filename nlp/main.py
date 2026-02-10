import json
from explain_transfer import explain_transfer
from explain_manufacturing import explain_manufacturing
from explain_scenario import explain_scenario

with open("sample_inputs/transfer.json") as f:
    print(explain_transfer(json.load(f)))

with open("sample_inputs/manufacturing.json") as f:
    print(explain_manufacturing(json.load(f)))

with open("sample_inputs/scenario.json") as f:
    print(explain_scenario(json.load(f)))
