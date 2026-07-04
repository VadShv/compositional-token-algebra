"""Repetition-heavy text samples (code, JSON logs, multi-turn chat with repeated system prompt)."""

CODE = '''import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

class Decoder(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return x

def train_step(model, x, optimizer):
    optimizer.zero_grad()
    out = model(x)
    loss = F.mse_loss(out, x)
    loss.backward()
    optimizer.step()
    return loss.item()
'''

LOGS = '''{"level": "INFO", "service": "auth", "event": "login", "user_id": 1001, "status": "ok"}
{"level": "INFO", "service": "auth", "event": "login", "user_id": 1002, "status": "ok"}
{"level": "ERROR", "service": "auth", "event": "login", "user_id": 1003, "status": "fail"}
{"level": "INFO", "service": "auth", "event": "logout", "user_id": 1001, "status": "ok"}
{"level": "INFO", "service": "auth", "event": "login", "user_id": 1004, "status": "ok"}
{"level": "ERROR", "service": "auth", "event": "login", "user_id": 1005, "status": "fail"}
{"level": "INFO", "service": "auth", "event": "login", "user_id": 1006, "status": "ok"}
'''

SYS = "You are a helpful assistant. Answer concisely and cite sources when possible."
CHAT = "\n".join([
    f"System: {SYS}\nUser: What is the capital of France?\nAssistant: Paris.",
    f"System: {SYS}\nUser: What is the capital of Japan?\nAssistant: Tokyo.",
    f"System: {SYS}\nUser: What is the capital of Italy?\nAssistant: Rome.",
    f"System: {SYS}\nUser: What is the capital of Spain?\nAssistant: Madrid.",
])

# A near-prose control (low repetition) to show CTA does nothing harmful when no repeats
PROSE = ('The history of the printing press begins in the fifteenth century when a '
         'goldsmith devised a method of movable type that transformed the spread of '
         'knowledge across the European continent within a few short decades.')

SAMPLES = {"code": CODE, "logs": LOGS, "chat": CHAT, "prose": PROSE}
