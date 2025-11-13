import os
import asyncio
from pathlib import Path
from prisma import Prisma
from dotenv import load_dotenv

# Fix: Use relative import from app.core
from template_tags import process_resumes_in_folder

load_dotenv()


async def seed_tags(db: Prisma, predefined_tags: list[str]) -> dict[str, str]:
    """
    Ensures all predefined tags exist in the database.
    Returns a mapping of tag names to their IDs.
    """
    tag_map = {}
    
    for tag_name in predefined_tags:
        # Upsert: create if doesn't exist, otherwise get existing
        tag = await db.tag.upsert(
            where={"name": tag_name},
            data={
                "create": {"name": tag_name},
                "update": {}
            }
        )
        tag_map[tag_name] = tag.id
        print(f"✓ Tag ensured: {tag_name} (ID: {tag.id})")
    
    return tag_map


async def seed_resume_templates(db: Prisma, processed_data: list[dict], tag_map: dict[str, str]):
    """
    Seeds the resume templates with their associated tags.
    """
    for item in processed_data:
        file_path = Path(item['filepath'])
        template_name = file_path.stem  # Filename without extension
        latex_content = item['content']
        tags = item['tags']
        
        # Fix: Use find_first instead of find_unique since name isn't unique
        existing = await db.resumetemplate.find_first(
            where={"name": template_name}
        )
        
        if existing:
            print(f"⚠ Template '{template_name}' already exists, skipping...")
            continue
        
        # Create the template with associated tags
        try:
            template = await db.resumetemplate.create(
                data={
                    "name": template_name,
                    "latexSrc": latex_content,
                    "previewImage": None,  # You can add preview image URLs later
                    "tags": {
                        "connect": [{"id": tag_map[tag]} for tag in tags if tag in tag_map]
                    }
                }
            )
            print(f"✓ Created template: {template_name} with tags: {tags}")
        
        except Exception as e:
            print(f"✗ Failed to create template '{template_name}': {e}")


async def main():
    """
    Main seeding function that orchestrates the entire process.
    """
    # Step 1: Process all resume templates and generate tags
    print("=" * 60)
    print("STEP 1: Processing resume templates and generating tags...")
    print("=" * 60)
    
    script_file_path = Path(__file__).resolve()
    project_root = script_file_path.parent.parent.parent
    FOLDER_TO_PROCESS = project_root / "templates"
    
    processed_data = process_resumes_in_folder(str(FOLDER_TO_PROCESS))
    
    if not processed_data:
        print("No templates were processed. Exiting.")
        return
    
    print(f"\n✓ Processed {len(processed_data)} templates\n")
    
    # Step 2: Connect to database
    print("=" * 60)
    print("STEP 2: Connecting to database...")
    print("=" * 60)
    
    db = Prisma()
    await db.connect()
    print("✓ Database connected\n")
    
    try:
        # Step 3: Seed tags
        print("=" * 60)
        print("STEP 3: Seeding tags...")
        print("=" * 60)
        
        PREDEFINED_TAGS = [
            "Technology",
            "Business & Finance",
            "Engineering",
            "Design & User Experience",
            "Freelance",
            "Corporate",
            "Startup",
            "Software Development"
        ]
        
        tag_map = await seed_tags(db, PREDEFINED_TAGS)
        print(f"\n✓ {len(tag_map)} tags seeded\n")
        
        # Step 4: Seed resume templates
        print("=" * 60)
        print("STEP 4: Seeding resume templates...")
        print("=" * 60)
        
        await seed_resume_templates(db, processed_data, tag_map)
        
        print("\n" + "=" * 60)
        print("✓ Database seeding completed successfully!")
        print("=" * 60)
        
    finally:
        await db.disconnect()
        print("\n✓ Database disconnected")


if __name__ == "__main__":
    asyncio.run(main())