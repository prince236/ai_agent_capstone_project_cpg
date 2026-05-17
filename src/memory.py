class Memory:
    def __init__(self):
        self.history = []

    def add(self, role, message):
        self.history.append({"role": role, "content": message})

    def get(self):
        return "\n".join(
            [f"{h['role']}: {h['content']}" for h in self.history]
        )
