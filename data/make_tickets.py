"""Generate realistic Jira-style structured tickets (repeated field templates)."""
import os, random, json
random.seed(0)

_CORPUS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "corpus")
os.makedirs(_CORPUS_DIR, exist_ok=True)

PROJECTS = ["AUTH", "PAY", "SEARCH", "INFRA", "MOBILE"]
TYPES = ["Bug", "Story", "Task", "Epic"]
PRIOS = ["Low", "Medium", "High", "Critical"]
STATUS = ["Open", "In Progress", "In Review", "Done"]
USERS = ["a.ivanov", "m.chen", "s.patel", "k.mueller", "t.tanaka"]
COMPONENTS = ["backend", "frontend", "database", "api", "auth-service"]
SUMMARIES = [
    "Login fails intermittently under high load",
    "Add pagination to search results endpoint",
    "Refactor payment retry logic for idempotency",
    "Dashboard latency spikes on first render",
    "Migrate session store to Redis cluster",
    "Fix null pointer in token refresh handler",
    "Implement rate limiting on public API",
    "Update dependency to patch CVE",
]

def ticket(i):
    proj = random.choice(PROJECTS)
    return "\n".join([
        f"Ticket: {proj}-{1000+i}",
        f"Type: {random.choice(TYPES)}",
        f"Priority: {random.choice(PRIOS)}",
        f"Status: {random.choice(STATUS)}",
        f"Assignee: {random.choice(USERS)}",
        f"Reporter: {random.choice(USERS)}",
        f"Component: {random.choice(COMPONENTS)}",
        f"Summary: {random.choice(SUMMARIES)}",
        f"Description: This issue was observed in the {random.choice(COMPONENTS)} component "
        f"and needs to be addressed before the next release. Steps to reproduce are documented.",
        "---",
    ])

tickets = "\n".join(ticket(i) for i in range(60))
open(os.path.join(_CORPUS_DIR, "tickets.txt"), "w").write(tickets)
print("tickets bytes:", len(tickets))
print(tickets[:400])
