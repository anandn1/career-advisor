from prisma import Prisma
from contextlib import asynccontextmanager

# Global Prisma client instance
prisma = Prisma()

async def connect_db():
    """Connect to the database"""
    if not prisma.is_connected():
        await prisma.connect()

async def disconnect_db():
    """Disconnect from the database"""
    if prisma.is_connected():
        await prisma.disconnect()

@asynccontextmanager
async def get_db():
    """Get database session context manager"""
    await connect_db()
    try:
        yield prisma
    finally:
        pass  # Keep connection alive for the app lifecycle
