"""
Prisma Models Usage Examples for Career Advisor

With Prisma, models are auto-generated from schema.prisma.
This file contains usage examples and helper functions.
"""

from typing import Optional, List
from prisma import Prisma
from .database import prisma


# ============================================
# Resume Template Operations
# ============================================

async def create_resume_template(
    name: str,
    latex_src: str,
    description: Optional[str] = None,
    preview_image: Optional[str] = None,
    category: Optional[str] = None,
    difficulty_level: Optional[str] = None,
    industry: Optional[str] = None,
    tag_ids: Optional[List[int]] = None,
):
    """Create a new resume template with optional tags"""
    tag_connections = [{"id": tag_id} for tag_id in tag_ids] if tag_ids else []
    
    template = await prisma.resumetemplate.create(
        data={
            "name": name,
            "latexSrc": latex_src,
            "description": description,
            "previewImage": preview_image,
            "category": category,
            "difficultyLevel": difficulty_level,
            "industry": industry,
            "tags": {"connect": tag_connections} if tag_connections else None,
        },
        include={"tags": True}
    )
    return template


async def get_template_by_id(template_id: int):
    """Get a resume template by ID with its tags"""
    return await prisma.resumetemplate.find_unique(
        where={"id": template_id},
        include={"tags": True}
    )


async def get_templates_by_tag(tag_name: str):
    """Get all templates with a specific tag"""
    return await prisma.resumetemplate.find_many(
        where={
            "tags": {
                "some": {"name": tag_name}
            }
        },
        include={"tags": True}
    )


async def get_templates_by_category(category: str):
    """Get templates by category"""
    return await prisma.resumetemplate.find_many(
        where={"category": category},
        include={"tags": True},
        order={"downloadCount": "desc"}
    )


async def update_template_downloads(template_id: int):
    """Increment template download count"""
    return await prisma.resumetemplate.update(
        where={"id": template_id},
        data={"downloadCount": {"increment": 1}}
    )


# ============================================
# Tag Operations
# ============================================

async def create_tag(name: str, color: Optional[str] = None):
    """Create a new tag"""
    return await prisma.tag.create(
        data={"name": name, "color": color}
    )


async def get_all_tags():
    """Get all tags with template count"""
    return await prisma.tag.find_many(
        include={"templates": True}
    )


# ============================================
# User Operations
# ============================================

async def create_user(
    email: str,
    full_name: str,
    hashed_password: str,
    current_role: Optional[str] = None,
):
    """Create a new user"""
    return await prisma.user.create(
        data={
            "email": email,
            "fullName": full_name,
            "hashedPassword": hashed_password,
            "currentRole": current_role,
        }
    )


async def get_user_by_email(email: str):
    """Get user by email"""
    return await prisma.user.find_unique(
        where={"email": email},
        include={"skills": True, "assessments": True}
    )


async def add_skill_to_user(user_id: int, skill_id: int):
    """Add a skill to a user"""
    return await prisma.user.update(
        where={"id": user_id},
        data={"skills": {"connect": [{"id": skill_id}]}}
    )


# ============================================
# Career Recommendation Operations
# ============================================

async def create_recommendation(
    user_id: int,
    career_path_id: int,
    match_score: float,
    reasoning: Optional[str] = None,
):
    """Create a career recommendation"""
    return await prisma.careerrecommendation.create(
        data={
            "userId": user_id,
            "careerPathId": career_path_id,
            "matchScore": match_score,
            "reasoning": reasoning,
        },
        include={"careerPath": True}
    )


async def get_user_recommendations(user_id: int):
    """Get all recommendations for a user"""
    return await prisma.careerrecommendation.find_many(
        where={"userId": user_id},
        include={"careerPath": True},
        order={"matchScore": "desc"}
    )
