# Prisma Setup Guide for Career Advisor

## Prerequisites
- Python 3.9+
- PostgreSQL installed and running
- pip

## Installation Steps

### 1. Install Dependencies
```powershell
cd backend
pip install -r requirements.txt
```

### 2. Configure Database
Create a `.env` file in the `backend` directory:
```env
DATABASE_URL="postgresql://username:password@localhost:5432/career_advisor?schema=public"
```

Replace `username`, `password`, and database name with your PostgreSQL credentials.

### 3. Generate Prisma Client
```powershell
prisma generate
```

This generates the Python client from your schema.

### 4. Push Schema to Database
```powershell
# For development - pushes schema without migrations
prisma db push

# OR for production - create migrations
prisma migrate dev --name init
```

### 5. (Optional) Seed the Database
Create sample data in `backend/prisma/seed.py`:
```python
from prisma import Prisma
import asyncio

async def seed():
    prisma = Prisma()
    await prisma.connect()
    
    # Create tags
    modern_tag = await prisma.tag.create(
        data={"name": "modern", "color": "#2196F3"}
    )
    
    # Create template
    await prisma.resumetemplate.create(
        data={
            "name": "Modern Professional",
            "latexSrc": "\\documentclass{article}...",
            "category": "professional",
            "tags": {"connect": [{"id": modern_tag.id}]}
        }
    )
    
    await prisma.disconnect()

if __name__ == "__main__":
    asyncio.run(seed())
```

Run seed:
```powershell
python prisma/seed.py
```

## Usage in FastAPI

### Update main.py:
```python
from fastapi import FastAPI
from app.db.database import connect_db, disconnect_db

app = FastAPI()

@app.on_event("startup")
async def startup():
    await connect_db()

@app.on_event("shutdown")
async def shutdown():
    await disconnect_db()
```

### Query Examples:
```python
from app.db import models

# Create template
template = await models.create_resume_template(
    name="Modern Tech Resume",
    latex_src="\\documentclass{article}...",
    category="professional",
    tag_ids=[1, 2, 3]
)

# Get templates by category
templates = await models.get_templates_by_category("professional")

# Get template by ID
template = await models.get_template_by_id(1)
```

## Prisma Commands

```powershell
# Generate client
prisma generate

# Push schema changes (dev)
prisma db push

# Create migration
prisma migrate dev --name migration_name

# Apply migrations (production)
prisma migrate deploy

# Open Prisma Studio (GUI)
prisma studio

# Format schema
prisma format

# Reset database (⚠️ deletes all data)
prisma migrate reset
```

## Troubleshooting

### "prisma command not found"
Install globally or use:
```powershell
python -m prisma generate
python -m prisma db push
```

### Import errors
Make sure to generate the client:
```powershell
prisma generate
```

### Connection errors
- Check PostgreSQL is running
- Verify DATABASE_URL in .env
- Ensure database exists: `CREATE DATABASE career_advisor;`

## Prisma vs SQLAlchemy Benefits

✅ **Type-safe queries** - Auto-completion in VS Code  
✅ **Automatic migrations** - Schema changes tracked  
✅ **Cleaner syntax** - Less boilerplate code  
✅ **Built-in async** - Native async/await support  
✅ **Prisma Studio** - Visual database browser  
✅ **Better relations** - Easier nested queries
