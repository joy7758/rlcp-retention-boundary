import json


class FLOPsLogger:
    def __init__(self):
        self.records = []

    def log(self, step, flops):
        self.records.append({"step": int(step), "flops": float(flops)})

    def total(self):
        return float(sum(item["flops"] for item in self.records))

    def payload(self):
        return {
            "total_flops": self.total(),
            "num_records": len(self.records),
            "records": self.records,
        }

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.payload(), f, indent=2)
