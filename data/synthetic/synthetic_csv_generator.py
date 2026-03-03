import pandas as pd
import random
import json
from faker import Faker
from datetime import datetime, timedelta

fake = Faker()

num_rows = 1000
rows = []

departments = ["Finance", "HR", "IT", "Operations", "Marketing", "Sales", "Support"]
systems = ["Email Server", "Database", "Payroll System", "Network Switch", "Web Portal", "CRM"]
priority_levels = ["Low", "Medium", "High", "Critical"]

for i in range(num_rows):
    # Hierarchical headers: Ticket / Customer / Resolution
    ticket_id = f"TCKT-{1000+i}"
    created_date = fake.date_between(start_date='-2y', end_date='today')
    date_formats = [
        created_date.strftime("%Y-%m-%d"), 
        created_date.strftime("%d/%m/%y"), 
        created_date.strftime("%b %d, %Y")
    ]
    created = random.choice(date_formats)
    
    resolved_date = created_date + timedelta(days=random.randint(0, 30))
    resolved = resolved_date.strftime(random.choice(["%Y-%m-%d", "%d/%m/%y", "%b %d, %Y"]))
    
    customer_name = fake.name()
    customer_email = fake.email()
    
    priority = random.choice(priority_levels)
    
    affected = [{"system": random.choice(systems), "impact": random.choice(["Partial", "Full"])} 
                for _ in range(random.randint(1,3))]
    affected_json = json.dumps(affected)
    
    actions = [{"step": fake.sentence(nb_words=6), "performed_by": fake.name()} 
               for _ in range(random.randint(1,3))]
    actions_json = json.dumps(actions)
    
    impacted_departments = ";".join(random.sample(departments, k=random.randint(1,3)))
    
    description = fake.paragraph(nb_sentences=3)
    if random.random() < 0.3:
        description += " <br> " + fake.word() + "!"
    
    if random.random() < 0.1:
        description = ""
    if random.random() < 0.05:
        customer_email = None
    
    row = [
        ticket_id, created, resolved, priority, impacted_departments,
        customer_name, customer_email,
        affected_json, actions_json, description
    ]
    
    rows.append(row)

# Hierarchical headers with Ticket as top-level
header = pd.MultiIndex.from_tuples([
    ("Ticket", "Ticket ID"),
    ("Ticket", "Created Date"),
    ("Ticket", "Resolved Date"),
    ("Ticket", "Priority"),
    ("Ticket", "Impacted Departments"),
    ("Customer", "Name"),
    ("Customer", "Email"),
    ("Resolution", "Affected Systems"),
    ("Resolution", "Actions Taken"),
    ("Resolution", "Description")
])

df = pd.DataFrame(rows, columns=header)

# Optional duplicate column
df[("Resolution", "Description Copy")] = df[("Resolution", "Description")]

# Save CSV
df.to_csv("professional_incident_tickets.csv", index=False, encoding="utf-8")
print("CSV generated: professional_incident_tickets.csv")